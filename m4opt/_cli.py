import shlex
import sys
from contextlib import contextmanager
from itertools import accumulate, chain
from typing import Annotated, Iterable, cast

import click
import numpy as np
import typer
from astropy import units as u
from astropy.coordinates import ICRS
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization.units import quantity_support as _quantity_support
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from ligo.skymap.plot.allsky import AutoScaledWCSAxes
from ligo.skymap.plot.poly import cut_prime_meridian
from ligo.skymap.postprocess import find_greedy_credible_levels
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.transforms import BlendedAffine2D
from matplotlib.typing import ColorType

from . import missions
from .fov import footprint, footprint_healpix
from .milp import Model
from .utils.console import progress, status
from .utils.dynamics import nominal_roll
from .utils.numpy import arange_with_units, clump_nonzero

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def prime():
    """Download and cache all dependencies that m4opt may use at runtime.

    Under normal operation, m4opt will download and cache various external
    data sources (for example, IERS Earth orientation data and Planck dust
    maps). If you need to run m4opt in an environment with no outbound Internet
    connectivity (for example, some computing clusters), you can run this
    command to download and cache the external data sources immediately.
    """
    from .models._extinction import dust_map

    dust_map()


class MissionParam(click.Choice):
    def __init__(self):
        choices = [name for name in missions.__all__ if name[0].islower()]
        super().__init__(choices)

    def convert(self, value, *args, **kwargs):
        if isinstance(value, missions.Mission):
            return value
        else:
            return getattr(missions, super().convert(value, *args, **kwargs))


MissionOption = Annotated[
    missions.Mission, typer.Option(click_type=MissionParam(), show_default=False)
]


def invert_footprints(footprints, n_pixels):
    pixels_to_fields_map = [[] for _ in range(n_pixels)]
    for i, js in enumerate(footprints):
        for j in js:
            pixels_to_fields_map[j].append(i)
    return [
        np.asarray(field_indices, dtype=np.intp)
        for field_indices in pixels_to_fields_map
    ]


@app.command()
@progress()
@u.quantity_input
def schedule(
    skymap: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Sky map filename", metavar="INPUT.multiorder.fits"),
    ],
    schedule: Annotated[
        typer.FileTextWrite,
        typer.Argument(
            help="Output filename for generated schedule", metavar="SCHEDULE.ecsv"
        ),
    ],
    mission: MissionOption = missions.uvex,
    delay: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Delay from time of event until the start of observations",
        ),
    ] = "0 day",
    deadline: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Maximum time from event until the end of observations",
        ),
    ] = "1 day",
    time_step: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Time step for evaluating field of regard",
        ),
    ] = "1 min",
    exptime: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Exposure time for each observation",
        ),
    ] = "900 s",
    timelimit: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Time limit for MILP solver",
        ),
    ] = "1e75 s",
    nside: Annotated[int, typer.Option(help="Default HEALPix resolution")] = 512,
    jobs: Annotated[
        int,
        typer.Option(
            help="Number of threads for parallel processing, or 0 for all cores",
        ),
    ] = 0,
):
    """Schedule a target of opportunity observation."""
    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        event_time = Time(
            Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso"
        )

    with status("propagating orbit"):
        obstimes = event_time + arange_with_units(
            delay, deadline + time_step, time_step
        )
        observer_locations = mission.orbit(obstimes).earth_location

    with status("evaluating field of regard"):
        target_coords = mission.skygrid
        exptime_s = exptime.to_value(u.s)
        obstimes_s = (obstimes - obstimes[0]).to_value(u.s)
        observable_intervals = np.asarray(
            [
                obstimes_s[intervals]
                for intervals in clump_nonzero(
                    np.logical_and.reduce(
                        [
                            constraint(
                                observer_locations[:-1],
                                target_coords[:, np.newaxis],
                                obstimes[:-1],
                            )
                            for constraint in mission.constraints
                        ],
                        axis=0,
                    )
                )
            ],
            dtype=object,
        )

        # Subtract off exposure times from end times.
        for intervals in observable_intervals:
            intervals[:, 1] -= exptime_s

        # Keep only intervals that are at least as long as the exposure time.
        observable_intervals = np.asarray(
            [
                intervals[intervals[:, 1] - intervals[:, 0] >= 0]
                for intervals in observable_intervals
            ],
            dtype=object,
        )

        # Discard fields that are not observable.
        good = np.asarray([len(intervals) > 0 for intervals in observable_intervals])
        observable_intervals = observable_intervals[good]
        target_coords = target_coords[good]

        # Find the start and end times that a field is observable.
        start_time_lbs, start_time_ubs = np.transpose(
            [[intervals[0, 0], intervals[-1, -1]] for intervals in observable_intervals]
        )

    with status("calculating footprints"):
        # Compute nominal roll angles for optimal solar power.
        # The nominal roll angle varies as a function of sky position and time.
        # We compute it for the start of the schedule because we assume that it
        # does not change much over the duration.
        rolls = nominal_roll(observer_locations[0], target_coords, event_time)
        footprints = footprint_healpix(
            hpx,
            mission.fov,
            target_coords,
            rolls,
        )

        # Select only the most probable 100 fields.
        n_fields = 100
        if len(target_coords) > n_fields:
            good = np.argpartition(
                [-probs[footprint].sum() for footprint in footprints], n_fields
            )[:n_fields]
            target_coords = target_coords[good]
            rolls = rolls[good]
            footprints = footprints[good]
            start_time_lbs = start_time_lbs[good]
            start_time_ubs = start_time_ubs[good]
            observable_intervals = observable_intervals[good]
        else:
            n_fields = len(target_coords)

        # # Throw away pixels that are not contained in any fields.
        good = np.unique(np.concatenate(footprints))
        imap = np.empty(len(probs), dtype=np.intp)
        imap[good] = np.arange(len(good))
        probs = probs[good]
        footprints = np.asarray(
            [imap[footprint] for footprint in footprints], dtype=object
        )
        n_pixels = len(probs)

        pixels_to_fields_map = invert_footprints(footprints, n_pixels)

    with status("calculating slew times"):
        slew_i, slew_j = np.triu_indices(n_fields, 1)
        timediff_s = (
            exptime
            + mission.slew.time(
                target_coords[slew_i],
                target_coords[slew_j],
                rolls[slew_i],
                rolls[slew_j],
            )
        ).to_value(u.s)

    with status("assembling MILP model"):
        model = Model(timelimit=timelimit, jobs=jobs)
        pixel_vars = model.binary_vars(n_pixels)
        field_vars = model.binary_vars(n_fields)
        start_time_vars = model.continuous_vars(
            n_fields,
            lb=start_time_lbs,
            ub=start_time_ubs,
        )

        # Add constraints on observability windows for each field
        with status("adding field of regard constraints"):
            for start_time_var, intervals in zip(start_time_vars, observable_intervals):
                if len(intervals) > 1:
                    interval_vars = model.binary_vars(len(intervals))
                    model.add_sos1(interval_vars)
                    begin, end = intervals.T
                    model.add_indicator(interval_vars, start_time_var >= begin)
                    model.add_indicator(interval_vars, start_time_var <= end)

        with status("adding slew constraints"):
            model.add_constraints_(
                model.abs(start_time_vars[slew_i] - start_time_vars[slew_j])
                >= timediff_s * (field_vars[slew_i] + field_vars[slew_j] - 1)
            )

        with status("adding coverage constraints"):
            model.add_constraints_(
                pixel_vars
                <= [
                    model.sum_vars_all_different(field_vars[field_indices])
                    for field_indices in pixels_to_fields_map
                ]
            )

        with status("adding objective function"):
            model.maximize(model.scal_prod_vars_all_different(pixel_vars, probs))

    with status("solving MILP model"):
        solution = model.solve()

    with status("writing results"):
        if solution is None:
            field_values = np.zeros(n_fields, dtype=bool)
            start_time_values = np.zeros(n_fields)
            objective_value = 0.0
        else:
            field_values = np.asarray(solution.get_values(field_vars), dtype=bool)
            start_time_values = np.asarray(solution.get_values(start_time_vars))
            objective_value = solution.get_objective_value()

        table = QTable(
            {
                "start_time": obstimes[0] + start_time_values * u.s,
                "end_time": obstimes[0] + start_time_values * u.s + exptime,
                "target_coord": target_coords,
                "roll": rolls,
            },
            descriptions={
                "start_time": "Start time of observation",
                "end_time": "End time of observation",
                "target_coord": "Coordinates of the center of the FOV",
                "roll": "Position angle of the FOV",
            },
            meta={
                "command": shlex.join(sys.argv),
                "args": {
                    "deadline": deadline,
                    "delay": delay,
                    "mission": mission.name,
                    "nside": nside,
                    "time_step": time_step,
                    "skymap": skymap.name,
                },
                "objective_value": objective_value,
                "best_bound": model.solve_details.best_bound,
                "solution_status": model.solve_details.status,
                "solution_time": model.solve_details.time * u.s,
            },
        )[field_values]
        table.add_column(
            mission.orbit(table["start_time"]).earth_location,
            index=2,
            name="observer_location",
        )
        table["observer_location"].info.description = "Position of the spacecraft"
        table.sort("start_time")
        table.write(schedule, format="ascii.ecsv", overwrite=True)


@contextmanager
def quantity_support(*args, **kwargs):
    """Workaround for https://github.com/astropy/astropy/pull/17006."""
    with _quantity_support(*args, **kwargs):
        yield


@app.command()
@progress()
@quantity_support()
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
        u.Quantity[u.physical.time],
        typer.Option(
            parser=u.Quantity,
            help="Time step for evaluating field of regard",
        ),
    ] = "1 hour",
):
    with status("loading schedule"):
        table = QTable.read(schedule, format="ascii.ecsv")
        deadline = table.meta["args"]["deadline"]
        delay = table.meta["args"]["delay"]
        nside = table.meta["args"]["nside"]
        skymap = table.meta["args"]["skymap"]

    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_moc["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        event_time = Time(
            Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso"
        )

    with status("making animation"):
        with status("setting up axes"):
            fig = plt.figure()
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

            ax_map = fig.add_subplot(gs[0], projection="astro hours mollweide")
            assert isinstance(ax_map, AutoScaledWCSAxes)
            transform = ax_map.get_transform("world")
            ax_map.add_artist(
                ax_map.legend(
                    [
                        Patch(facecolor=footprint_color),
                        Patch(facecolor="none", edgecolor=skymap_color),
                    ],
                    [
                        "observation footprints",
                        "90% credible region",
                    ],
                    loc="upper left",
                    borderaxespad=0,
                    bbox_to_anchor=[0, 1],
                    bbox_transform=fig.transFigure,
                )
            )
            ax_map.legend(
                [
                    Patch(facecolor=averaged_field_of_regard_color),
                    Patch(
                        facecolor=field_of_regard_color,
                    ),
                ],
                [
                    "averaged",
                    "instantaneous",
                ],
                loc="upper right",
                title="outside field of regard",
                borderaxespad=0,
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
            ax_map.contour_hpx(cls, levels=[0.9], colors=[skymap_color], nested=True)
            ax_map.grid()

            observer_locations = mission.orbit(time_steps).earth_location

        with status("adding field of regard"):
            instantaneous_field_of_regard = np.logical_and.reduce(
                [
                    constraint(
                        observer_locations[:, np.newaxis],
                        hpx.healpix_to_skycoord(np.arange(hpx.npix)),
                        time_steps[:, np.newaxis],
                    )
                    for constraint in mission.constraints
                ],
                axis=0,
            )
            averaged_field_of_regard = np.logical_or.reduce(
                instantaneous_field_of_regard, axis=0
            )
            ax_map.contourf_hpx(
                averaged_field_of_regard,
                nested=True,
                levels=[-1, 0.5],
                colors=[averaged_field_of_regard_color],
                zorder=1.1,
            )

        with status("adding observation footprints"):
            footprint_patches = [
                [
                    plt.Polygon(
                        np.rad2deg(vertices),
                        transform=transform,
                        visible=False,
                        facecolor=footprint_color,
                    )
                    for vertices in cut_prime_meridian(
                        np.column_stack(
                            (region.vertices.ra.rad, region.vertices.dec.rad)
                        )
                    )
                ]
                for region in footprint(
                    mission.fov, table["target_coord"], table["roll"]
                )
            ]
            for patch in chain.from_iterable(footprint_patches):
                ax_map.add_patch(patch)

            table["area"], table["prob"] = zip(
                *map(
                    lambda pixels: [len(pixels) * hpx.pixel_area, probs[pixels].sum()],
                    accumulate(
                        footprint_healpix(
                            hpx, mission.fov, table["target_coord"], table["roll"]
                        ),
                        lambda *args: np.unique(np.concatenate(args)),
                    ),
                )
            )

            ax_timeline = fig.add_subplot(gs[1])
            ax_timeline.set_xlim(0 * u.hour, deadline)
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

            ax_timeline.step(
                (table["end_time"] - time_steps[0]).to(u.hour),
                table["prob"],
                color=skymap_color,
                where="post",
            )
            ax_area.step(
                (table["end_time"] - time_steps[0]).to(u.hour),
                table["area"],
                color=footprint_color,
                where="post",
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
            ax_timeline.axvspan(0 * u.hour, delay, color="lightgray")
            ax_timeline.text(
                delay,
                0.5,
                "delay",
                rotation=90,
                rotation_mode="anchor",
                horizontalalignment="center",
                verticalalignment="bottom",
                transform=BlendedAffine2D(ax_timeline.transData, ax_timeline.transAxes),
            )

        with status("rendering frames"):
            field_of_regard_artist = None

            def draw_frame(args):
                time, field_of_regard = args
                blit = [now_line, now_label]
                time_from_start = (time - event_time).to(u.hour)
                now_line.set_xdata([time_from_start])
                now_label.set_x(time_from_start)
                for patches, visible in zip(
                    footprint_patches, table["start_time"] <= time
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

                return blit

            FuncAnimation(
                fig,
                draw_frame,
                frames=zip(time_steps, instantaneous_field_of_regard),
                save_count=len(time_steps),
                blit=True,
            ).save(output.name)
