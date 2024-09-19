import shlex
import sys
from typing import Annotated

import numpy as np
import typer
from astropy import units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy_healpix import HEALPix
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map

from .. import __version__, missions
from ..fov import footprint_healpix
from ..milp import Model
from ..utils.console import progress, status
from ..utils.dynamics import nominal_roll
from ..utils.numpy import arange_with_units, clump_nonzero, full_indices
from .core import MissionOption, app


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
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 2,
    cadence: Annotated[
        u.Quantity[u.physical.time],
        typer.Option(parser=u.Quantity, help="Minimum time separation between visits"),
    ] = "30 min",
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
            min=0,
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
        # FIXME: https://github.com/astropy/astropy/issues/17030
        target_coords = SkyCoord(target_coords.ra, target_coords.dec)
        exptime_s = exptime.to_value(u.s)
        cadence_s = cadence.to_value(u.s)
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
        footprints = footprint_healpix(hpx, mission.fov, target_coords, rolls)

        # Select only the most probable 50 fields.
        n_fields = 50
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
        start_time_field_visit_vars = model.continuous_vars(
            (n_fields, visits),
            lb=np.tile(start_time_lbs[:, np.newaxis], visits),
            ub=np.tile(start_time_ubs[:, np.newaxis], visits),
        )

        # Add constraints on observability windows for each field
        with status("adding field of regard constraints"):
            for start_time_visit_vars, intervals in zip(
                start_time_field_visit_vars, observable_intervals
            ):
                if len(intervals) > 1:
                    begin, end = intervals.T
                    visit_interval_vars = model.binary_vars((visits, len(intervals)))
                    for interval_vars in visit_interval_vars:
                        model.add_sos1(interval_vars)
                    model.add_indicators(
                        visit_interval_vars,
                        start_time_visit_vars[:, np.newaxis] >= begin,
                    )
                    model.add_indicators(
                        visit_interval_vars, start_time_visit_vars[:, np.newaxis] <= end
                    )

        with status("adding cadence constraints"):
            model.add_constraints_(
                start_time_field_visit_vars[:, 1:] - start_time_field_visit_vars[:, :-1]
                >= cadence_s * field_vars
            )

        with status("adding slew constraints"):
            p, q = full_indices(visits)
            model.add_constraints_(
                model.abs(
                    start_time_field_visit_vars[slew_i, p[:, np.newaxis]]
                    - start_time_field_visit_vars[slew_j, q[:, np.newaxis]]
                )
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
            field_values = np.zeros(field_vars.shape, dtype=bool)
            start_time_field_visit_values = np.empty(start_time_field_visit_vars.shape)
            objective_value = 0.0
        else:
            field_values = np.asarray(solution.get_values(field_vars), dtype=bool)
            start_time_field_visit_values = np.asarray(
                solution.get_values(start_time_field_visit_vars.ravel())
            ).reshape(start_time_field_visit_vars.shape)
            objective_value = solution.get_objective_value()

        table = QTable(
            {
                "action": np.full(field_values.sum() * visits, "observe"),
                "start_time": obstimes[0]
                + start_time_field_visit_values[field_values].ravel() * u.s,
                "duration": np.full(field_values.sum() * visits, exptime.value)
                * exptime.unit,
                "target_coord": target_coords[
                    np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                ].ravel(),
                "roll": rolls[
                    np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                ].ravel(),
            },
            descriptions={
                "action": "Action for the spacecraft",
                "start_time": "Start time of segment",
                "duration": "Duration of segment",
                "target_coord": "Coordinates of the center of the FOV",
                "roll": "Position angle of the FOV",
            },
            meta={
                "command": shlex.join(sys.argv),
                "version": __version__,
                "args": {
                    "deadline": deadline,
                    "delay": delay,
                    "mission": mission.name,
                    "nside": nside,
                    "time_step": time_step,
                    "skymap": skymap.name,
                    "visits": visits,
                },
                "objective_value": objective_value,
                "best_bound": model.solve_details.best_bound,
                "solution_status": model.solve_details.status,
                "solution_time": model.solve_details.time * u.s,
            },
        )
        table.sort("start_time")

        # Add orbit to table
        table.add_column(
            mission.orbit(table["start_time"]).earth_location,
            index=3,
            name="observer_location",
        )
        table["observer_location"].info.description = "Position of the spacecraft"

        # Add slew segments to table
        nrows = len(table) - 1
        slew_table = QTable(
            {
                "action": np.full(nrows, "slew"),
                "start_time": (table["start_time"] + table["duration"])[:-1],
                "duration": mission.slew.time(
                    table["target_coord"][:-1],
                    table["target_coord"][1:],
                    table["roll"][:-1],
                    table["roll"][1:],
                ),
                # FIXME: dummy values.
                # See https://github.com/astropy/astropy/issues/14292
                "target_coord": SkyCoord(
                    np.zeros(nrows) * u.deg, np.zeros(nrows) * u.deg
                ),
            }
        )
        table = vstack(
            (
                table,
                slew_table,
            )
        )
        table.sort("start_time")

        # Calculate total time spent observing, slewing, etc.,
        # as well as the amount of unused slack time
        total_time_by_action = (
            table["action", "duration"].group_by("action").groups.aggregate(np.sum)
        )
        table.meta["total_time"] = {
            str(row["action"]): row["duration"] for row in total_time_by_action
        }
        table.meta["total_time"]["slack"] = (
            deadline - delay - total_time_by_action["duration"].sum()
        )

        table.write(schedule, format="ascii.ecsv", overwrite=True)
