from itertools import accumulate, chain
from pathlib import Path
from typing import Annotated, Iterable, cast

import numpy as np
import synphot
import typer
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord, get_body
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization.units import quantity_support
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot.allsky import AutoScaledWCSAxes
from ligo.skymap.plot.marker import earth, moon, sun
from ligo.skymap.plot.poly import cut_prime_meridian
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Patch
from matplotlib.transforms import BlendedAffine2D
from matplotlib.typing import ColorType

from .. import missions
from ..fov import footprint, footprint_healpix
from ..synphot import observing
from ..synphot.extinction import DustExtinction
from ..utils.console import progress, status
from .core import app


@app.command()
@progress()
def animate(
    schedule: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Input filename for schedule", metavar="SCHEDULE.ecsv"),
    ],
    output: Annotated[
        typer.FileBinaryWrite,
        typer.Argument(help="Output filename for animation", metavar="MOVIE.gif"),
    ],
    time_step: Annotated[
        u.Quantity,
        typer.Option(
            help="Time step for evaluating field of regard",
        ),
    ] = 1 * u.hour,
    duration: Annotated[
        u.Quantity,
        typer.Option(
            help="Duration of animation",
        ),
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
        typer.Option(
            help="Radius of optional zoomed inset",
        ),
    ] = 10 * u.deg,
):
    """Generate an animation for a GW sky map."""
    with status("loading schedule"):
        table = QTable.read(schedule, format="ascii.ecsv")
        table["end_time"] = table["start_time"] + table["duration"]
        table = table[table["action"] == "observe"]
        deadline = table.meta["args"]["deadline"]
        delay = table.meta["args"]["delay"]
        nside = table.meta["args"]["nside"]
        skymap = table.meta["args"]["skymap"]
        visits = table.meta["args"]["visits"]
        absmag_mean = table.meta["args"]["absmag_mean"]
        bandpass = table.meta["args"]["bandpass"]
        snr = table.meta["args"]["snr"]
        exptime_min = table.meta["args"]["exptime_min"]

    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        event_time = Time(
            Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso"
        )

    with status("making animation"), quantity_support():
        with status("setting up axes"):
            fig = plt.figure(dpi=dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
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
            footprint_alpha = 1 / visits

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
                    dict(marker=earth, mec="black"),
                    dict(marker=sun, mec="black"),
                    dict(marker=moon(-115), mfc="black", mec="none"),
                ]
            ]
            fig.legend(
                [
                    Patch(facecolor=footprint_color, alpha=footprint_alpha),
                    Patch(facecolor="none", edgecolor=skymap_color),
                ],
                [
                    "observation footprints",
                    "90% credible region",
                ],
                loc="outside upper left",
                bbox_to_anchor=[0, 1],
                bbox_transform=fig.transFigure,
            )
            fig.legend(
                [
                    Patch(facecolor=averaged_field_of_regard_color),
                    Patch(facecolor=field_of_regard_color),
                ],
                [
                    "averaged",
                    "instantaneous",
                ],
                loc="outside upper right",
                title="outside field of regard",
                bbox_to_anchor=[1, 1],
                bbox_transform=fig.transFigure,
            )

            mission: missions.Mission = getattr(missions, table.meta["args"]["mission"])
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

            observer_locations = mission.observer_location(time_steps)

        if absmag_mean is not None:
            if mission.detector is None:
                raise NotImplementedError(
                    "This mission does not define a detector model"
                )
            with status("adding exposure time map"):
                distmod = Distance(skymap_moc.meta["distmean"] * u.Mpc).distmod
                with observing(
                    observer_location=observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(np.arange(hpx.npix)),
                    obstime=time_steps[0],
                ):
                    exptime = mission.detector.get_exptime(
                        snr,
                        synphot.SourceSpectrum(
                            synphot.ConstFlux1D(absmag_mean * u.ABmag + distmod)
                        )
                        * DustExtinction(),
                        bandpass,
                    ).to_value(u.s)
                ims = [
                    ax.imshow_hpx(
                        exptime,
                        vmin=table["duration"].min(initial=exptime_min).to_value(u.s),
                        vmax=table["duration"].max(initial=exptime_min).to_value(u.s),
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

        with status("adding field of regard"):
            earth_coords, sun_coords, moon_coords = [
                get_body(body, time_steps, observer_locations)
                for body in ["earth", "sun", "moon"]
            ]

            instantaneous_field_of_regard = mission.constraints(
                observer_locations[:, np.newaxis],
                hpx.healpix_to_skycoord(np.arange(hpx.npix)),
                time_steps[:, np.newaxis],
            )
            averaged_field_of_regard = np.logical_or.reduce(
                instantaneous_field_of_regard, axis=0
            )
            for ax in ax_maps:
                ax.contourf_hpx(
                    averaged_field_of_regard,
                    nested=True,
                    levels=[-1, 0.5],
                    colors=[averaged_field_of_regard_color],
                    zorder=1.1,
                )

        with status("adding observation footprints"):
            footprint_regions = footprint(
                mission.fov, table["target_coord"], table["roll"]
            )
            footprint_patches = [
                [
                    plt.Polygon(
                        np.rad2deg(vertices),
                        transform=transforms[0],
                        visible=False,
                        facecolor=footprint_color,
                        alpha=footprint_alpha,
                    )
                    for vertices in cut_prime_meridian(
                        np.column_stack(
                            (region.vertices.ra.rad, region.vertices.dec.rad)
                        )
                    )
                ]
                for region in footprint_regions
            ]
            for patch in chain.from_iterable(footprint_patches):
                ax_map.add_patch(patch)
            if inset_center is not None:
                footprint_patches_zoom = [
                    plt.Polygon(
                        np.column_stack(
                            (region.vertices.ra.deg, region.vertices.dec.deg)
                        ),
                        transform=transforms[1],
                        visible=False,
                        facecolor=footprint_color,
                        alpha=footprint_alpha,
                    )
                    for region in footprint_regions
                ]
                for patch in footprint_patches_zoom:
                    ax_map_zoom.add_patch(patch)

            ivisit = np.arange(visits)
            table["area"] = np.empty((len(table), visits)) * hpx.pixel_area.to(u.deg**2)
            table["prob"] = np.empty((len(table), visits))
            for row, selected_pixels in zip(
                table,
                accumulate(
                    footprint_healpix(
                        hpx, mission.fov, table["target_coord"], table["roll"]
                    ),
                    lambda *args: np.concatenate(args),
                ),
            ):
                visited = (
                    np.bincount(selected_pixels, minlength=hpx.npix)
                    > ivisit[:, np.newaxis]
                )
                row["area"] = np.count_nonzero(visited, axis=1) * hpx.pixel_area.to(
                    u.deg**2
                )
                row["prob"] = np.sum(probs * visited, axis=1)

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

            for i, (area, prob) in enumerate(zip(table["area"].T, table["prob"].T)):
                alpha = (i + 1) / visits
                visit_skymap_color, visit_footprint_color = np.asarray(
                    [to_rgb(color) for color in [skymap_color, footprint_color]]
                ) * alpha + (1 - alpha)
                ax_timeline.hlines(
                    prob,
                    (table["start_time"] - time_steps[0]).to(u.hour),
                    (table["end_time"] - time_steps[0]).to(u.hour),
                    visit_skymap_color,
                )
                ax_area.hlines(
                    area,
                    (table["start_time"] - time_steps[0]).to(u.hour),
                    (table["end_time"] - time_steps[0]).to(u.hour),
                    visit_footprint_color,
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
            if delay != 0 * u.hour:
                ax_timeline.axvspan(0 * u.hour, delay, color="lightgray")
                ax_timeline.text(
                    delay,
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

                for patches, visible in zip(
                    footprint_patches, table["start_time"] <= time
                ):
                    for patch in patches:
                        if visible != patch.get_visible():
                            patch.set_visible(visible)
                            blit.append(patch)
                if inset_center is not None:
                    for patch, visible in zip(
                        footprint_patches_zoom, table["start_time"] <= time
                    ):
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
