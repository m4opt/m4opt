import shlex
import sys
from typing import Annotated

import click
import numpy as np
import typer
from astropy import units as u
from astropy.table import QTable
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map

from . import missions
from .fov import footprint_healpix
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
    skymap: Annotated[typer.FileBinaryRead, typer.Argument(help="Sky map filename")],
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
            help="Number of threads for parallel processing, or 0 to use a number of threads that is appropriate for the number of CPUs that you have"
        ),
    ] = 0,
):
    """Schedule a ToO observation."""
    with status("loading sky map"):
        hpx = HEALPix(nside, order="nested")
        skymap_table = read_sky_map(skymap, moc=True)
        probs = rasterize(skymap_table["UNIQ", "PROBDENSITY"], hpx.level)["PROB"]
        event_time = Time(
            Time(skymap_table.meta["gps_time"], format="gps").utc, format="iso"
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
            nside,
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
            for field_var, start_time_var, intervals in zip(
                field_vars, start_time_vars, observable_intervals
            ):
                if len(intervals) > 1:
                    interval_vars = model.binary_vars(len(intervals))
                    begin, end = intervals.T
                    model.add_constraint_(
                        field_var >= model.sum_vars_all_different(interval_vars)
                    )
                    for interval_var, interval in zip(interval_vars, intervals):
                        begin, end = interval
                        model.add_indicator(interval_var, start_time_var >= begin)
                        model.add_indicator(interval_var, start_time_var <= end)

        # Add no overlap constraints
        with status("adding no overlap constraints"):
            sequence_vars = model.binary_vars(n_fields * (n_fields - 1) // 2)
            s = sequence_vars
            for i, (fi, ti) in enumerate(zip(field_vars, start_time_vars)):
                model.add_indicators(
                    s[: n_fields - i - 1],
                    (ti + exptime_s * fi <= tj for tj in start_time_vars[i + 1 :]),
                    true_values=1,
                )
                model.add_indicators(
                    s[: n_fields - i - 1],
                    (
                        tj + exptime_s * fj <= ti
                        for fj, tj in zip(field_vars[i + 1 :], start_time_vars[i + 1 :])
                    ),
                    true_values=0,
                )
                s = s[n_fields - i - 1 :]

        with status("adding coverage constraints"):
            model.add_constraints_(
                pixel_var <= model.sum_vars_all_different(field_vars[field_indices])
                for pixel_var, field_indices in zip(pixel_vars, pixels_to_fields_map)
            )

        with status("adding objective function"):
            model.maximize(model.scal_prod_vars_all_different(pixel_vars, probs))

    with status("solving MILP model"):
        solution = model.solve()

    with status("writing results"):
        if solution is None:
            field_values = np.zeros(n_fields, dtype=bool)
            objective_value = 0.0
        else:
            field_values = np.asarray(solution.get_values(field_vars), dtype=bool)
            objective_value = solution.get_objective_value()

        start_time_values = np.asarray(solution.get_values(start_time_vars))

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
        table.write("example.ecsv", overwrite=True)
