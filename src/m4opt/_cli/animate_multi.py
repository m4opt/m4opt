from collections.abc import Iterable
from dataclasses import dataclass
from itertools import accumulate, chain
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import synphot
import typer
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord, get_body
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization.units import quantity_support
from astropy_healpix import HEALPix
from click import UsageError
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot.allsky import AutoScaledWCSAxes
from ligo.skymap.plot.marker import earth, moon, sun
from ligo.skymap.plot.poly import cut_prime_meridian
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from matplotlib.transforms import BlendedAffine2D
from matplotlib.typing import ColorType
from regions import Regions

from .. import missions
from ..fov import footprint, footprint_healpix
from ..synphot import observing
from ..synphot.extinction import DustExtinction
from ..utils.console import progress, status
from .core import app


def _joint_coverage_steps(
    probabilities: np.ndarray,
    events: Iterable[tuple[float, np.ndarray]],
    event_time: Time,
) -> list[tuple[float, float]]:
    """Return cumulative union probability after each observation starts."""
    covered_pixels = np.zeros(len(probabilities), dtype=bool)
    event_hours = sorted(
        [
            ((start_gps - event_time.gps) / 3600, pixels)
            for start_gps, pixels in events
        ],
        key=lambda event: event[0],
    )
    steps = []
    i = 0
    while i < len(event_hours):
        hour = event_hours[i][0]
        while i < len(event_hours) and event_hours[i][0] == hour:
            covered_pixels[event_hours[i][1]] = True
            i += 1
        steps.append((hour, float(np.sum(probabilities[covered_pixels]))))
    return steps


@dataclass
class Schedule:
    name: str
    table: QTable
    mission: missions.Mission
    visits: int
    footprint_patches: list[list[Polygon]]
    footprint_patches_zoom: list[list[Polygon]]


@app.command()
@progress()
def animate_multi(
    schedules: Annotated[
        list[typer.FileBinaryRead],
        typer.Argument(help="Input schedule files to compare", metavar="SCHEDULE.ecsv"),
    ],
    output: Annotated[
        typer.FileBinaryWrite,
        typer.Option("--output", "-o", help="Output filename for animation"),
    ],
    time_step: Annotated[
        u.Quantity,
        typer.Option(help="Time step for evaluating field of regard"),
    ] = 10 * u.min,
    duration: Annotated[
        u.Quantity,
        typer.Option(help="Duration of animation"),
    ] = 5 * u.s,
    still: Annotated[
        typer.FileBinaryWrite | None,
        typer.Option(help="Optional output file for still frame", metavar="STILL.pdf"),
    ] = None,
    dpi: Annotated[
        float | None,
        typer.Option(
            help="Output resolution [default: Matplotlib default setting]",
            show_default=False,
        ),
    ] = None,
    inset_center: Annotated[
        SkyCoord | None,
        typer.Option(help="Center of optional zoomed inset", parser=SkyCoord),
    ] = None,
    inset_radius: Annotated[
        u.Quantity,
        typer.Option(help="Radius of optional zoomed inset"),
    ] = 10 * u.deg,
):
    """Generate one animation comparing any number of schedules."""
    if len(schedules) < 2:
        raise UsageError("animate-multi requires at least two schedule files.")

    with status("loading schedules"):
        schedule_data: list[Schedule] = []
        for source in schedules:
            table = QTable.read(source, format="ascii.ecsv")
            table["end_time"] = table["start_time"] + table["duration"]
            table = table[table["action"] == "observe"]
            args = table.meta["args"]
            schedule_data.append(
                Schedule(
                    name=Path(source.name).stem,
                    table=table,
                    mission=getattr(missions, args["mission"]),
                    visits=args["visits"],
                    footprint_patches=[],
                    footprint_patches_zoom=[],
                )
            )

        first_args = schedule_data[0].table.meta["args"]
        for schedule in schedule_data[1:]:
            args = schedule.table.meta["args"]
            for key in ("skymap", "nside", "event_time"):
                if args.get(key) != first_args.get(key):
                    raise UsageError(
                        "All schedules must use the same sky map, nside, and "
                        f"event time; mismatch in {key!r}."
                    )

        deadline = max(
            (schedule.table.meta["args"]["deadline"] for schedule in schedule_data),
            key=lambda value: value.to_value(u.s),
        )
        delays = [schedule.table.meta["args"]["delay"] for schedule in schedule_data]
        common_delay = all(delay == delays[0] for delay in delays[1:])
        nside = first_args["nside"]
        skymap = first_args["skymap"]
        event_time = first_args.get("event_time")

    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        if event_time is None:
            try:
                gps_time = skymap_moc.meta["gps_time"]
            except KeyError:
                raise UsageError(
                    f'Neither the schedules nor the sky map "{skymap}" says what '
                    "time the schedules were measured from."
                ) from None
            event_time = Time(Time(gps_time, format="gps").utc, format="iso")
        else:
            event_time = Time(event_time)

    with status("making animation"), quantity_support():
        with status("setting up axes"):
            fig = plt.figure(dpi=dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
            schedule_colors = plt.get_cmap("plasma")(
                np.linspace(0.05, 0.95, len(schedule_data))
            )
            colormap = plt.get_cmap("Paired")
            assert isinstance(colormap, ListedColormap)
            (
                field_of_regard_color,
                averaged_field_of_regard_color,
                _,
                skymap_color,
                _,
                footprint_color,
                *_,
            ) = cast(Iterable[ColorType], colormap.colors)
            now_color = "black"

            ax_map = fig.add_subplot(gs[0], projection="astro hours mollweide")
            assert isinstance(ax_map, AutoScaledWCSAxes)
            ax_maps = [ax_map]
            if inset_center is not None:
                ax_map_zoom = fig.add_axes(
                    (0.75, 0.45, 0.25, 0.25),
                    projection="astro hours zoom",
                    center=inset_center,
                    radius=inset_radius,
                )
                assert isinstance(ax_map_zoom, AutoScaledWCSAxes)
                for key in ["ra", "dec"]:
                    ax_map_zoom.coords[key].set_ticklabel_visible(False)
                    ax_map_zoom.coords[key].set_ticks_visible(False)
                ax_map.mark_inset_axes(ax_map_zoom, zorder=10000)
                for loc in ["upper left", "lower left"]:
                    ax_map.connect_inset_axes(ax_map_zoom, loc, zorder=10000)
                ax_maps.append(ax_map_zoom)
            transforms = [ax.get_transform("world") for ax in ax_maps]

            earth_artists, sun_artists, moon_artists = [
                [
                    ax.plot(
                        0,
                        0,
                        zorder=10000,
                        transform=transform,
                        **kwargs,
                    )[0]
                    for ax, transform in zip(ax_maps, transforms)
                ]
                for kwargs in [
                    {"marker": earth, "mec": "black"},
                    {"marker": sun, "mec": "black"},
                    {"marker": moon(-115), "mfc": "black", "mec": "none"},
                ]
            ]

            fig.legend(
                handles=[
                    Patch(facecolor=color, label=schedule.name)
                    for schedule, color in zip(schedule_data, schedule_colors)
                ],
                loc="outside upper left",
                bbox_to_anchor=[0, 1],
                bbox_transform=fig.transFigure,
                title="schedules",
            )
            fig.legend(
                [
                    Patch(facecolor=averaged_field_of_regard_color),
                    Patch(facecolor=field_of_regard_color),
                    Patch(facecolor="none", edgecolor=skymap_color),
                ],
                [
                    "averaged field of regard",
                    "instantaneous field of regard",
                    "90% credible region",
                ],
                loc="outside upper right",
                bbox_to_anchor=[1, 1],
                bbox_transform=fig.transFigure,
            )

            time_steps = (
                event_time
                + np.arange(
                    0,
                    (deadline + time_step).to_value(u.s),
                    time_step.to_value(u.s),
                )
                * u.s
            )
            cls = find_greedy_credible_levels(probs)
            for ax in ax_maps:
                ax.contour_hpx(cls, levels=[0.9], colors=[skymap_color], nested=True)
            ax_map.grid()

            observer_locations = [
                schedule.mission.observer_location(time_steps)
                for schedule in schedule_data
            ]
            absmag_mean = first_args["absmag_mean"]
            if absmag_mean is not None:
                for schedule in schedule_data[1:]:
                    args = schedule.table.meta["args"]
                    for key in ("mission", "absmag_mean", "bandpass", "snr"):
                        if args.get(key) != first_args.get(key):
                            raise UsageError(
                                "Exposure-time maps require matching mission, "
                                f"magnitude, bandpass, and SNR; mismatch in {key!r}."
                            )
                mission = schedule_data[0].mission
                if mission.detector is None:
                    raise NotImplementedError(
                        "This mission does not define a detector model"
                    )
                bandpass = first_args["bandpass"]
                if isinstance(bandpass, str) or bandpass is None:
                    bandpass = [bandpass]
                if len(set(bandpass)) > 1:
                    raise NotImplementedError(
                        "An exposure time map is not supported for a schedule that "
                        "uses more than one bandpass."
                    )
                with status("adding exposure time map"):
                    distmod = Distance(skymap_moc.meta["distmean"] * u.Mpc).distmod
                    with observing(
                        observer_location=observer_locations[0][0],
                        target_coord=hpx.healpix_to_skycoord(
                            np.arange(hpx.npix)
                        ),
                        obstime=time_steps[0],
                    ):
                        exptime = mission.detector.get_exptime(
                            first_args["snr"],
                            synphot.SourceSpectrum(
                                synphot.ConstFlux1D(absmag_mean * u.ABmag + distmod)
                            )
                            * DustExtinction(),
                            bandpass[0],
                        ).to_value(u.s)
                    exptime_min = u.Quantity(first_args["exptime_min"]).min()
                    vmin = min(
                        schedule.table["duration"].min(initial=exptime_min)
                        for schedule in schedule_data
                    )
                    vmax = max(
                        schedule.table["duration"].max(initial=exptime_min)
                        for schedule in schedule_data
                    )
                    ims = [
                        ax.imshow_hpx(
                            exptime,
                            vmin=vmin.to_value(u.s),
                            vmax=vmax.to_value(u.s),
                            cmap="binary",
                            nested=True,
                            alpha=0.5,
                        )
                        for ax in ax_maps
                    ]
                    plt.colorbar(
                        ims[0],
                        label="Required exposure time (s)",
                        orientation="horizontal",
                    ).ax.xaxis.set_label_position("top")

            joint_events: list[tuple[float, np.ndarray]] = []
            for schedule, color in zip(schedule_data, schedule_colors):
                table = schedule.table
                footprint_regions = footprint(
                    schedule.mission.fov, table["target_coord"], table["roll"]
                )
                footprint_patches = [
                    [
                        plt.Polygon(
                            np.rad2deg(vertices),
                            transform=transforms[0],
                            visible=False,
                            facecolor=color,
                            alpha=0.8,
                        )
                        for subregion in (
                            region.regions if isinstance(region, Regions) else [region]
                        )
                        for vertices in cut_prime_meridian(
                            np.column_stack(
                                (subregion.vertices.ra.rad, subregion.vertices.dec.rad)
                            )
                        )
                    ]
                    for region in footprint_regions
                ]
                for patch in chain.from_iterable(footprint_patches):
                    ax_map.add_patch(patch)
                schedule.footprint_patches = footprint_patches

                if inset_center is not None:
                    footprint_patches_zoom = [
                        [
                            plt.Polygon(
                                np.column_stack(
                                    (
                                        subregion.vertices.ra.deg,
                                        subregion.vertices.dec.deg,
                                    )
                                ),
                                transform=transforms[1],
                                visible=False,
                                facecolor=color,
                                alpha=0.8,
                            )
                            for subregion in (
                                region.regions
                                if isinstance(region, Regions)
                                else [region]
                            )
                        ]
                        for region in footprint_regions
                    ]
                    for patch in chain.from_iterable(footprint_patches_zoom):
                        ax_map_zoom.add_patch(patch)
                    schedule.footprint_patches_zoom = footprint_patches_zoom

                ivisit = np.arange(schedule.visits)
                table["area"] = (
                    np.empty((len(table), schedule.visits))
                    * hpx.pixel_area.to(u.deg**2)
                )
                table["prob"] = np.empty((len(table), schedule.visits))
                selected_by_observation = list(
                    footprint_healpix(
                        hpx,
                        schedule.mission.fov,
                        table["target_coord"],
                        table["roll"],
                    )
                )
                for row, selected_pixels in zip(
                    table,
                    accumulate(
                        selected_by_observation,
                        lambda *args: np.concatenate(args),
                    ),
                ):
                    visited = (
                        np.bincount(selected_pixels, minlength=hpx.npix)
                        > ivisit[:, np.newaxis]
                    )
                    row["area"] = (
                        np.count_nonzero(visited, axis=1) * hpx.pixel_area.to(u.deg**2)
                    )
                    row["prob"] = np.sum(probs * visited, axis=1)

                joint_events.extend(
                    (float(start_time.gps), pixels)
                    for start_time, pixels in zip(
                        table["start_time"], selected_by_observation
                    )
                )

            ax_timeline = fig.add_subplot(gs[1])
            ax_timeline.set_xlim(0 * u.hour, deadline.to(u.hour))
            ax_timeline.set_xlabel(f"time in hours since event at {event_time}")
            ax_timeline.set_ylim(0, 1)
            ax_timeline.yaxis.set_tick_params(
                color=skymap_color, labelcolor=skymap_color
            )
            ax_timeline.set_ylabel("probability", color=skymap_color)

            ax_area = ax_timeline.twinx()
            ax_area.yaxis.set_tick_params(
                color=footprint_color, labelcolor=footprint_color
            )
            ax_area.set_ylabel("area (deg$^2$)", color=footprint_color)
            joint_probability_color = "black"
            ax_timeline.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        color=joint_probability_color,
                        linewidth=2.5,
                        label="joint probability",
                    )
                ],
                loc="upper left",
            )

            for schedule, color in zip(schedule_data, schedule_colors):
                table = schedule.table
                rgb = np.asarray(to_rgb(color))
                for area, prob in zip(table["area"].T, table["prob"].T):
                    ax_timeline.hlines(
                        prob,
                        (table["start_time"] - time_steps[0]).to(u.hour),
                        (table["end_time"] - time_steps[0]).to(u.hour),
                        rgb,
                        alpha=0.55,
                    )
                    ax_area.hlines(
                        area,
                        (table["start_time"] - time_steps[0]).to(u.hour),
                        (table["end_time"] - time_steps[0]).to(u.hour),
                        rgb,
                    )

            previous_hour = 0.0
            previous_probability = 0.0
            deadline_hours = deadline.to_value(u.hour)
            for hour, probability in _joint_coverage_steps(
                probs, joint_events, event_time
            ):
                if hour > deadline_hours:
                    break
                hour = max(hour, 0)
                if hour > previous_hour:
                    ax_timeline.hlines(
                        previous_probability,
                        previous_hour * u.hour,
                        hour * u.hour,
                        color=joint_probability_color,
                        linewidth=2.5,
                        zorder=5,
                    )
                if probability != previous_probability:
                    ax_timeline.vlines(
                        hour * u.hour,
                        previous_probability,
                        probability,
                        color=joint_probability_color,
                        linewidth=2.5,
                        zorder=5,
                    )
                previous_hour = hour
                previous_probability = probability
            if deadline_hours > previous_hour:
                ax_timeline.hlines(
                    previous_probability,
                    previous_hour * u.hour,
                    deadline,
                    color=joint_probability_color,
                    linewidth=2.5,
                    zorder=5,
                )
            ax_area.set_ylim(0 * u.deg**2)

            now_line = ax_timeline.axvline(
                (time_steps[0] - event_time).to(u.hour),
                color=now_color,
            )
            now_label = ax_timeline.text(
                (time_steps[0] - event_time).to(u.hour),
                1,
                "now",
                ha="center",
                va="bottom",
                color=now_color,
                transform=BlendedAffine2D(ax_timeline.transData, ax_timeline.transAxes),
            )
            if common_delay and delays[0] != 0 * u.hour:
                ax_timeline.axvspan(0 * u.hour, delays[0], color="lightgray")
                ax_timeline.text(
                    delays[0],
                    0.5,
                    "delay",
                    rotation=90,
                    rotation_mode="anchor",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    transform=BlendedAffine2D(
                        ax_timeline.transData, ax_timeline.transAxes
                    ),
                )

        with status("adding field of regard"):
            instant_fields = [
                schedule.mission.constraints(
                    locations[:, np.newaxis],
                    hpx.healpix_to_skycoord(np.arange(hpx.npix)),
                    time_steps[:, np.newaxis],
                )
                for schedule, locations in zip(schedule_data, observer_locations)
            ]
            instantaneous_field_of_regard = np.logical_or.reduce(instant_fields)
            averaged_field_of_regard = np.any(instantaneous_field_of_regard, axis=0)
            for ax in ax_maps:
                ax.contourf_hpx(
                    averaged_field_of_regard,
                    nested=True,
                    levels=[-1, 0.5],
                    colors=[averaged_field_of_regard_color],
                    zorder=1.1,
                )

            earth_coords, sun_coords, moon_coords = [
                get_body(body, time_steps, observer_locations[0])
                for body in ["earth", "sun", "moon"]
            ]

        with status("rendering frames"):
            field_of_regard_artist = None
            field_of_regard_artist_zoom = None

            def draw_frame(args):
                time, field_of_regard, earth_coord, sun_coord, moon_coord = args
                blit = [
                    now_line,
                    now_label,
                    *earth_artists,
                    *sun_artists,
                    *moon_artists,
                ]

                time_from_start = (time - event_time).to(u.hour)
                now_line.set_xdata([time_from_start])
                now_label.set_x(time_from_start)

                for artists, coord in zip(
                    [earth_artists, sun_artists, moon_artists],
                    [earth_coord, sun_coord, moon_coord],
                ):
                    for artist in artists:
                        artist.set_data([[coord.ra.deg], [coord.dec.deg]])

                for schedule in schedule_data:
                    visible_rows = schedule.table["start_time"] <= time
                    for patches, visible in zip(
                        schedule.footprint_patches, visible_rows
                    ):
                        for patch in patches:
                            if visible != patch.get_visible():
                                patch.set_visible(visible)
                                blit.append(patch)
                    if inset_center is not None:
                        for patches, visible in zip(
                            schedule.footprint_patches_zoom, visible_rows
                        ):
                            for patch in patches:
                                if visible != patch.get_visible():
                                    patch.set_visible(visible)
                                    blit.append(patch)

                nonlocal field_of_regard_artist
                if field_of_regard_artist is not None:
                    blit.append(field_of_regard_artist)
                    field_of_regard_artist.remove()
                field_of_regard_artist = ax_map.contourf_hpx(
                    field_of_regard,
                    nested=True,
                    colors=[field_of_regard_color],
                    levels=[-1, 0.5],
                )
                blit.append(field_of_regard_artist)

                if inset_center is not None:
                    nonlocal field_of_regard_artist_zoom
                    if field_of_regard_artist_zoom is not None:
                        blit.append(field_of_regard_artist_zoom)
                        field_of_regard_artist_zoom.remove()
                    field_of_regard_artist_zoom = ax_map_zoom.contourf_hpx(
                        field_of_regard,
                        nested=True,
                        colors=[field_of_regard_color],
                        levels=[-1, 0.5],
                    )
                    blit.append(field_of_regard_artist_zoom)

                return blit

            FuncAnimation(
                fig,
                draw_frame,
                frames=zip(
                    time_steps,
                    instantaneous_field_of_regard,
                    earth_coords,
                    sun_coords,
                    moon_coords,
                ),
                save_count=len(time_steps),
                blit=True,
                interval=duration.to_value(u.ms) / len(time_steps),
            ).save(output.name)

        if still is not None:
            with status("saving still frame"):
                fig.savefig(still, format=Path(still.name).suffix.lstrip("."))
