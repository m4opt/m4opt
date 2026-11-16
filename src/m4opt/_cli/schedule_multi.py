import heapq
import shlex
import sys
from itertools import pairwise
from typing import Annotated
from pathlib import Path

import numpy as np
import synphot
import typer
import healpy as hp
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy_healpix import HEALPix
from click import UsageError
from docplex.mp.progress import ProgressData, ProgressDataRecorder
from ligo.skymap import distance
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from scipy import stats

from m4opt import __version__ 
from m4opt import missions as miss
from m4opt.dynamics import nominal_roll
from m4opt.fov import footprint_healpix
from m4opt.milp import Model
from m4opt.observer import EarthFixedObserverLocation
from m4opt.synphot import TabularScaleFactor, observing
from m4opt.synphot.extinction import DustExtinction
from m4opt.utils.console import progress, status
from m4opt.utils.numpy import clump_nonzero_inclusive, full_indices
from ._types import PhysicalQuantity
from m4opt._cli.core import app
from .schedule import (
    invert_footprints, 
    invert_footprints_to_regions, 
    prepare_piecewise_breakpoints,
)

'''
Implementation of a multiple missions scheduler.
Work in progress
'''

LARGE_EXPTIME = 1e10

def remap_footprints(footprints, pixel_map):
    return [pixel_map[footprint] for footprint in footprints]

def select_missions(mission_names):
    available_missions = {
        name: getattr(miss, name)
        for name in miss.__all__
        if name != "Mission"
        and isinstance(getattr(miss, name), miss.Mission)
    }
    unknown_missions = set(mission_names) - available_missions.keys()
    if unknown_missions:
        raise UsageError(
            f"Unknown mission(s): {', '.join(sorted(unknown_missions))}. "
            f"Choose from: {', '.join(available_missions)}."
        )
    if len(set(mission_names)) != len(mission_names):
        raise UsageError("Each mission may only be specified once.")
    if not mission_names:
        raise UsageError("Specify at least one mission with --mission.")
    return {name: available_missions[name] for name in mission_names}


def prepare_visit_configuration(visits, bandpass, exptime_min): # need modification because for now filter change is not really supported
    if visits < 1:
        raise UsageError("The number of visits must be at least one.")
    if not exptime_min:
        raise UsageError("Specify at least one minimum exposure time.")

    visit_bandpasses = [
        bandpass[i % len(bandpass)] if bandpass else None
        for i in range(visits)
    ]
    visit_exptime_min_s = u.Quantity(
        [exptime_min[i % len(exptime_min)] for i in range(visits)]
    ).to_value(u.s)
    return visit_bandpasses, visit_exptime_min_s

def build_mission_table(
    mission_name,
    mission_obj,                       # missions[mission_name]
    *,
    field_vars,                        # dict
    time_field_visit_vars,             # dict
    n_fields,                          # dict
    target_coords, rolls, field_ids,   # dicts
    visit_bandpasses,                  # dict
    solution,
    visits,
    visit_exptime_min_s,
    obstimes,
    deadline, delay,
    meta_common,                       # common val for all missions
    objective_value, model, has_model,
):
    nf = n_fields[mission_name]

    # per mission value extraction
    if solution is None:
        field_values = np.zeros(nf, dtype=bool)
        time_field_visit_values = np.empty((nf, visits))
        exptime_field_values = np.empty((nf, visits))
    else:
        field_values = solution.get_values(field_vars[mission_name]) >= 0.5
        time_field_visit_values = solution.get_values(time_field_visit_vars[mission_name])
        exptime_field_values = np.tile(visit_exptime_min_s, (nf, 1))

    # per mission table
    idx = np.flatnonzero(field_values)
    n_obs = idx.size * visits
    print("visit_exptime_min_s:", type(visit_exptime_min_s),
        getattr(visit_exptime_min_s, 'shape', None),
        visit_exptime_min_s
    )
    table = QTable(
        {
            "action": np.full(n_obs, "observe"),
            "start_time": obstimes[0]
                + (time_field_visit_values[field_values]
                   - 0.5 * exptime_field_values[field_values]).ravel() * u.s,
            "duration": exptime_field_values[field_values].ravel() * u.s,
            "target_coord": target_coords[mission_name][
                np.tile(idx[:, np.newaxis], visits)
            ].ravel(),
            "roll": rolls[mission_name][
                np.tile(idx[:, np.newaxis], visits)
            ].ravel(),
            "field_id": field_ids[mission_name][
                np.tile(idx[:, np.newaxis], visits)
            ].ravel(),
            "bandpass": np.tile(
                np.array([b or "" for b in visit_bandpasses], dtype=str),
                idx.size,
            ),
        },
        descriptions={
            "action": "Action for the spacecraft",
            "start_time": "Start time of segment",
            "duration": "Duration of segment",
            "target_coord": "Coordinates of the center of the FOV",
            "roll": "Position angle of the FOV",
            "field_id": "The mission's ID for the field observed",
            "bandpass": "Detector bandpass",
        },   # identiques
    )
    table.sort("start_time")

    # mission's intrinsic observer (not common)
    table.add_column(
        mission_obj.observer_location(table["start_time"]),
        index=3, name="observer_location",
    )
    table["observer_location"].info.description = "Position of the spacecraft"

    # slew
    if len(table) > 0:
        nrows = len(table) - 1
        slew_duration = mission_obj.slew.time(
            table["target_coord"][:-1], table["target_coord"][1:],
            table["roll"][:-1], table["roll"][1:],
        )
        changed = table["bandpass"][:-1] != table["bandpass"][1:]
        slew_duration[changed] = np.maximum(
            slew_duration[changed], mission_obj.filter_exchange_time
        )
        slew_table = QTable({
            "action": np.full(nrows, "slew"),
            "start_time": (table["start_time"] + table["duration"])[:-1],
            "duration": slew_duration,
            "bandpass": np.full(nrows, ""),
        })
        table = vstack((table, slew_table))
    table.sort("start_time")

    # mission's specific meta data
    if len(table) > 0:
        total_time_by_action = (
            table["action", "duration"].group_by("action").groups.aggregate(np.sum)
        )
        table.meta["total_time"] = {
            str(row["action"]): row["duration"].to(u.s) for row in total_time_by_action
        }
        table.meta["total_time"]["slack"] = (
            deadline - delay
            - total_time_by_action["duration"].sum()
        ).to(u.s)
    else:
        table.meta["total_time"] = {}

    table.meta.update({
        "command": shlex.join(sys.argv),
        "version": __version__,
        "args": {
            **meta_common,
            "mission": mission_obj.name,
            "bandpass": visit_bandpasses,
        },
        "objective_value": objective_value,
        "best_bound": model.best_bound if has_model else 0,
        "solution_status": (
            model.solve_details.status if has_model
            else "infeasible, no observable fields or pixels"
        ),
        "solution_time": (model.solve_details.time if has_model else 0) * u.s,
    })
    return table

@app.command()
@progress()
def schedule_multi(
    skymap: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Sky map filename", metavar="INPUT.multiorder.fits"),
    ],
    schedule: Annotated[
        str,
        typer.Argument(
            help="Output filename for generated schedule, attention, don't provide the extension, the scheduler will produce several out files (1 per mission)", metavar="SCHEDULE.ecsv"
        ),
    ],
    exptime_min: Annotated[
        list[PhysicalQuantity[u.physical.time]],
        typer.Option(
            help="Minimum exposure time for each observation. Repeat the "
            "option to give each bandpass its own exposure time, in the same "
            "order as --bandpass; a single value applies to every bandpass",
        ),
    ] = (30 * u.s,),
    skygrid: Annotated[
        str | None,
        typer.Option(
            help="Name of sky grid to use, if the mission supports multiple sky grids.",
        ),
    ] = None,
    event_time: Annotated[
        Time | None,
        typer.Option(
            help="Time of the event, which --delay and --deadline are measured "
            "from. Defaults to the DATE-OBS field in the sky map header.",
        ),
    ] = None,
    delay: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(
            help="Delay from time of event until the start of observations",
        ),
    ] = 0 * u.day,
    deadline: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(
            help="Maximum time from event until the end of observations",
        ),
    ] = 1 * u.day,
    time_step: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(
            help="Time step for evaluating field of regard",
        ),
    ] = 1 * u.min,
    exptime_max: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(
            help="Maximum exposure time for each observation",
        ),
    ] = np.inf * u.s,
    absmag_mean: Annotated[
        float | None,
        typer.Option(
            help="Mean AB absolute magnitude of source",
            show_default="disable adaptive exposure time",
        ),
    ] = None,
    absmag_stdev: Annotated[
        float,
        typer.Option(
            help="Standard deviation of AB absolute magnitude of source",
            show_default="AB absolute magnitude is fixed at the value provided by --absmag-mean",
        ),
    ] = 0.0,
    appmag_dist: Annotated[
        bool, typer.Option(help="Enable point-wise distribution of apparent magnitude")
    ] = True,
    snr: Annotated[float, typer.Option(help="Signal to noise ratio for detection")] = 5,
    bandpass: Annotated[
        list[str] | None,
        typer.Option(
            help="Name of detector bandpass. Repeat the option to cycle through "
            "several bandpasses on successive visits; for example, "
            "--bandpass g --bandpass r observes each field in g and then in r. "
            "Visits are grouped into contiguous blocks of a single bandpass so "
            "that the filter is exchanged only between blocks."
        ),
    ] = ['r'], # must be changed to have per facility filter !
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 1,
    cadence: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(help="Minimum time separation between visits"),
    ] = 30 * u.min,
    nside: Annotated[int, typer.Option(help="HEALPix resolution")] = 512,
    max_fields: Annotated[
        int,
        typer.Option(
            min=1,
            help="Consider only this many of the most probable fields. Raising "
            "it grows the problem roughly quadratically",
        ),
    ] = 50,
    overlap_weight: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            help="Weight for the overlap term in the objective function. 0=no overlap, 1=recount each pixel when preselecting the most probable fields",
        ),
    ] = 0.75,
    timelimit: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(
            help="Time limit for MILP solver",
            rich_help_panel="Solver Options",
        ),
    ] = 1 * u.h,
    memory: Annotated[
        PhysicalQuantity[u.physical.data_quantity],
        typer.Option(
            help="Maximum solver memory usage before terminating",
            rich_help_panel="Solver Options",
        ),
    ] = np.inf * u.GiB,
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            "-j",
            min=0,
            help="Number of threads for parallel processing, or 0 for all cores",
            rich_help_panel="Solver Options",
        ),
    ] = 0,
    cutoff: Annotated[
        float | None,
        typer.Option(
            min=0,
            max=1,
            help="Objective cutoff. Give up if there are no feasible solutions with objective value greater than or equal to this value",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    write_progress: Annotated[
        typer.FileTextWrite | None,
        typer.Option(
            help="Save a time series of the CPLEX objective value and best bound to this file",
            metavar="PROGRESS.ecsv",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    write_model: Annotated[
        typer.FileBinaryWrite | None,
        typer.Option(
            help="Export the MILP model in LP, SAV, or MPS format. Mainly useful for troubleshooting purposes",
            metavar="MODEL.{lp,mps,sav}[.gz]",
            rich_help_panel="Solver Options",
        ),
    ] = None,
    mission_names: Annotated[
        list[str],
        typer.Option(
            "--mission-names",
            help="Mission to include in the joint schedule. Repeat for each mission.",
        ),
    ] = ["rubin", "ztf"],
    write_candidates: Annotated[
        str | None,
        typer.Option(
            help="Save observable candidate fields and their HEALPix footprints for each mission to an NPZ file. The input is a suffix, the saved files will be named as <suffix>_<mission>.npz",
            metavar="<suffix>_<mission>.npz",
        ),
    ] = None,
):
    """
    Generate an observing plan for a (GW) skymap.

    This is a work in progress, major changes are expected.
    For now, only supports a fixed exposure time. 

    Planned changes:
    - Switch from a mission list approach to a Network approach (with a dedicated Network class)
    - Implement the other two modes of `m4opt schedule` (adaptative exposure time)

    # m4opt schedule-multi

    ### Current state: only a mono-filter, fixed exposure time version (adaptation of the [problem 1](https://m4opt.readthedocs.io/en/latest/design/milp.html#problem-1-fixed-exposure-time) from the documentation)
    ### Limitations:
    - No revisit: `visit`==1 ⇒ No cadence constraint
    - Unique filter implementation ⇒ No filter change constraint, for now, all missions share the same filter. I want to modify that use one filter that is mission specific

    ### Different preselection:
    The preselection of fields is now made jointly by computing the marginal probability gain (+ an overlap weight) of a tile using `heapq`. The overlap weight is a variable between 0 and 1 allowing more or less overlap between the tiles. With 0 overlap, no tiles will ever superpose (unlike in the classic scheduler) and with a wieght of 1, it mimicks the preselection of `m4opt schedule` (independant pre selection for each mission, hence causing a strong overlap between the different missions' pre selected tiles).

    ### Constraints:
    - FoV, slew, cuts exactly the same → **mission specific constraints grows linearly with the number of facilities**
    - Joint coverage constraints: `pixel_vars` is now shared across all facilities, concatenate all individual missions' `good pixels`. 

    ### Objective
    Same implementation (so $\max \sum_i \rho_i \, p_i$) but with

    $$p_i \le \sum_f \sum_{j \in J_i^{(f)}} m_{f,j} \quad \forall i$$

    With:
    - $\sum_f$ a sum over the different facilities (missions) considered
    - $m_{f,j}$ is the $j$-th field of facility $f$
    - $J_i^{(f)}$ is the set of indices of fields that contain the pixel $i$ for facility $f$

    **Note:** The only shared var important to the MILP problem is the pixels covered. Hence, pixel probability is only counted once even if multiple telescopes observe it.
    """
    missions = select_missions(mission_names)
    with status(f"Using missions: {missions.keys()}"):
        adaptive_exptime = False # force constant exp time for now

    visit_bandpasses, visit_exptime_min_s = prepare_visit_configuration(
        visits, bandpass, exptime_min
    )
    if adaptive_exptime and bandpass is not None and len(bandpass) > 1:
        raise NotImplementedError(
            "A variable exposure time is not supported with more than one bandpass."
        )
    filter_changes = [lhs != rhs for lhs, rhs in pairwise(visit_bandpasses)]

    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        skymap_flat = rasterize(skymap_moc, hpx.level)
        if event_time is None:
            # The sky map carries the trigger time unless one was given.
            try:
                gps_time = skymap_moc.meta["gps_time"]
            except KeyError:
                raise UsageError(
                    f'The sky map "{skymap.name}" has no DATE-OBS in its header, '
                    "which is where the time of the event is read from. "
                    "Pass --event-time instead."
                ) from None
            event_time = Time(Time(gps_time, format="gps").utc, format="iso")
        else:
            # Convert the user-supplied time to ISO/UTC so that it is consistent with the rest of the code
            event_time = Time(event_time, format="iso", scale="utc")

    with status("propagating orbit"):
        obstimes = event_time + np.arange(
            delay, deadline + time_step, time_step, like=time_step
        )
        observer_locations = {}
        for mission in missions.keys():
            observer_locations[mission] = missions[mission].observer_location(obstimes)

    with status("evaluating field of regard"):
        target_coords = {}
        field_ids = {}
        observable_intervals = {}
        for mission in missions.keys():
            if not isinstance(missions[mission].skygrid, dict):
                target_coords[mission] = missions[mission].skygrid
            elif skygrid in missions[mission].skygrid:
                target_coords[mission] = missions[mission].skygrid[skygrid]
            else:
                raise UsageError(
                    f"skygrid '{skygrid}' not found. Options: {', '.join(map(str, missions[mission].skygrid.keys()))}"
                )

            # The row of the grid is the mission's own name for the field. A
            # mission that numbers its fields leaves gaps, masked out of the grid
            # and dropped here so that everything below is dense.
            keep = ~target_coords[mission].mask
            field_ids[mission] = np.arange(len(target_coords[mission]))[keep]
            target_coords[mission] = target_coords[mission].unmasked[keep]
            # FIXME: https://github.com/astropy/astropy/issues/17030
            target_coords[mission] = SkyCoord(target_coords[mission].ra, target_coords[mission].dec)
            cadence_s = cadence.to_value(u.s) # now only fix cadence so no problem leaving it like this
            obstimes_s = (obstimes - obstimes[0]).to_value(u.s)
            observable_intervals[mission] = np.asarray(
                [
                    obstimes_s[intervals]
                    for intervals in clump_nonzero_inclusive(
                        missions[mission].constraints(
                            observer_locations[mission],
                            target_coords[mission][:, np.newaxis],
                            obstimes,
                        )
                    )
                ],
                dtype=object,
            )

            # Keep only intervals that are at least as long as the exposure time.
            exptime_min_s = visit_exptime_min_s.min()
            observable_intervals[mission] = np.asarray(
                [
                    intervals[intervals[:, 1] - intervals[:, 0] >= exptime_min_s]
                    for intervals in observable_intervals[mission]
                ],
                dtype=object,
            )

            # Discard fields that are not observable.
            good = np.asarray([len(intervals) > 0 for intervals in observable_intervals[mission]])
            observable_intervals[mission] = observable_intervals[mission][good]
            target_coords[mission] = target_coords[mission][good]
            field_ids[mission] = field_ids[mission][good]    

    with status("calculating footprints"):
        rolls = {}
        global_good_pixels = {} # try to have a unique pixel var for multiple mission
        footprints = {}
        n_fields = {}
        n_pixels = {}
        skymap_flat_full = skymap_flat
        all_footprints = {}
        for mission in missions.keys():
            if isinstance(missions[mission].observer_location, EarthFixedObserverLocation):
                rolls[mission] = np.zeros(len(target_coords[mission])) * u.deg
            else:
                # Compute nominal roll angles for optimal solar power.
                # The nominal roll angle varies as a function of sky position and time.
                # We compute it for the start of the schedule because we assume that it
                # does not change much over the duration.
                rolls[mission] = nominal_roll(observer_locations[mission][0], target_coords[mission], event_time)
            footprints[mission] = footprint_healpix(hpx, missions[mission].fov, target_coords[mission], rolls[mission])
            all_footprints[mission] = footprints[mission]

        pix_prob = np.asarray(skymap_flat_full["PROB"])
        counts = np.zeros(len(pix_prob), dtype=np.int32)

        selected = {mission: [] for mission in missions.keys()}

        def marginal_gain(fp):
            fp = np.asarray(fp)
            p = pix_prob[fp]
            is_new = counts[fp] == 0
            return p[is_new].sum() + overlap_weight * p[~is_new].sum() 

        heap = []
        for mission in missions.keys():
            for i, fp in enumerate(footprints[mission]):
                heapq.heappush(heap, (-marginal_gain(fp), mission, i)) 

        n_covered_version = 0                      
        gain_version = {}                          
        full = set()        

        while heap and len(full) < len(missions):
            neg_gain, mission, i = heapq.heappop(heap)
            if mission in full:
                continue
            if gain_version.get((mission, i), -1) != n_covered_version:
                gain_version[(mission, i)] = n_covered_version
                heapq.heappush(heap, (-marginal_gain(footprints[mission][i]), mission, i))
                continue
            if -neg_gain <= 0:
                break                            
            selected[mission].append(i)
            counts[np.asarray(footprints[mission][i])] += 1
            n_covered_version += 1
            if len(selected[mission]) >= max_fields:
                full.add(mission)
        print(f"Found best fields after {n_covered_version} iterations, covering {np.count_nonzero(counts)} pixels")
        for mission in missions.keys():
            good = np.array(sorted(selected[mission]), dtype=int)
            n_fields[mission] = len(good)
            target_coords[mission] = target_coords[mission][good]
            rolls[mission] = rolls[mission][good]
            footprints[mission] = footprints[mission][good]
            observable_intervals[mission] = observable_intervals[mission][good]
            field_ids[mission] = field_ids[mission][good]

            # save the candidates footprint for a later use
            if write_candidates is not None:
                write_candidates_mission = f"{write_candidates}_{mission}.npz"
                candidate_footprints = [
                    np.asarray(footprint, dtype=np.intp) for footprint in footprints[mission]
                ]
                footprint_lengths = np.fromiter(
                    (len(footprint) for footprint in candidate_footprints),
                    dtype=np.intp,
                    count=len(candidate_footprints),
                )
                footprint_offsets = np.concatenate(
                    (np.zeros(1, dtype=np.intp), np.cumsum(footprint_lengths))
                )
                candidate_pixels = (
                    np.concatenate(candidate_footprints)
                    if candidate_footprints
                    else np.asarray([], dtype=np.intp)
                )
                candidate_probability = np.asarray(
                    [
                        skymap_flat[footprint]["PROB"].sum()
                        for footprint in candidate_footprints
                    ]
                )
                np.savez(
                    write_candidates_mission,
                    field_id=field_ids[mission],
                    ra_deg=target_coords[mission].ra.to_value(u.deg),
                    dec_deg=target_coords[mission].dec.to_value(u.deg),
                    roll_deg=rolls[mission].to_value(u.deg),
                    probability=candidate_probability,
                    footprint_pixels=candidate_pixels,
                    footprint_offsets=footprint_offsets,
                    nside=nside,
                    order="nested",
                )

            # Throw away pixels that are not contained in any fields.
            good_pixels = (
                np.unique(np.concatenate(footprints[mission]))
                if len(footprints[mission]) > 0
                else np.asarray([], dtype=np.intp)
            )
            global_good_pixels[mission] = good_pixels
            imap = np.empty(len(skymap_flat_full), dtype=np.intp)
            imap[good_pixels] = np.arange(len(good_pixels))
            footprints[mission] = remap_footprints(footprints[mission], imap)
            n_pixels[mission] = len(good_pixels)

            #pixels_to_fields_map[mission] = invert_footprints(footprints[mission], n_pixels[mission]) -> useless ?

    with status("calculating slew times"):
        slew_time_s = {}
        slew_indices = {}
        for mission in missions.keys():
            slew_i, slew_j = np.triu_indices(n_fields[mission], 1)
            slew_indices[mission] = (slew_i, slew_j)
            slew_time_s[mission] = missions[mission].slew.time(
                target_coords[mission][slew_i],
                target_coords[mission][slew_j],
                rolls[mission][slew_i],
                rolls[mission][slew_j],
            ).to_value(u.s)

    # construct global pixel var 
    all_pixels_global = np.unique(
    np.concatenate([global_good_pixels[m] for m in missions])
    )
    if len(all_pixels_global) == 0:
        raise ValueError("No observable pixels")
    n_pixels_global = len(all_pixels_global)
    assert n_pixels_global >= max(n_pixels[m] for m in missions), \
        "n_pixels_global < max local : bug dans l'union" # AI assert as a failsafe but should never appear

    # remap HEALPIX index to the all pixel var 
    global_map = np.full(len(skymap_flat_full), -1, dtype=np.intp)
    global_map[all_pixels_global] = np.arange(n_pixels_global)

    skymap_flat_global = skymap_flat_full[all_pixels_global]

    # Remap each mission's footprint to the global footprint space
    footprints_global = {}
    pixels_to_fields_map_global = {}
    for mission in missions:
        local_to_global = global_map[global_good_pixels[mission]]
        footprints_global[mission] = remap_footprints(
            footprints[mission], local_to_global
        )
        assert all((f >= 0).all() for f in footprints_global[mission]), \
            f"missing pixel for the union of mission {mission}"
        pixels_to_fields_map_global[mission] = invert_footprints(
            footprints_global[mission], n_pixels_global
        )

    print("skymap_full PROB:",
        float(skymap_flat_full["PROB"].min()),
        float(skymap_flat_full["PROB"].max()),
        float(skymap_flat_full["PROB"].mean()),
        float(skymap_flat_full["PROB"].sum()))          # ← sum ≈ 1 ?
    print("skymap_full n_pixels:", len(skymap_flat_full))
    print("all_pixels_global min/max:", all_pixels_global.min(), all_pixels_global.max())
    print("frac PROB covered:",
        float(skymap_flat_full["PROB"][all_pixels_global].sum()
                / skymap_flat_full["PROB"].sum()))

    nside_skymap = hp.npix2nside(len(skymap_flat_full))
    ipix_peak = int(np.argmax(skymap_flat_full["PROB"]))
    ra_peak, dec_peak = hp.pix2ang(nside_skymap, ipix_peak, lonlat=True)
    print(f"skymap peak: RA={ra_peak:.2f} Dec={dec_peak:.2f} PROB={skymap_flat_full['PROB'][ipix_peak]:.3e}")

    for m in missions:
        tc = target_coords[m]
        if len(tc) == 0:
            print(f"{m}: n_obs=0, no fields are observable in the scheduling window")
            continue
        sep = tc.separation(SkyCoord(ra_peak*u.deg, dec_peak*u.deg)).deg
        i_best = int(np.argmin(sep))
        print(f"{m}: n_obs={len(tc)}, best field to peak = {sep[i_best]:.2f} deg, "
            f"field RA/Dec = ({tc[i_best].ra.deg:.2f}, {tc[i_best].dec.deg:.2f})")

    # do the MILP model
    with Model(
            timelimit=timelimit, jobs=jobs, memory=memory, lowercutoff=cutoff
        ) as model:
        with status("assembling MILP model"): 
            pixel_vars = model.binary_vars(n_pixels_global)

            field_vars = {}
            time_field_visit_vars = {}
            for mission in missions.keys():
                field_vars[mission] = model.binary_vars(n_fields[mission])
                time_field_visit_vars[mission] = model.continuous_vars(
                    (n_fields[mission], visits),
                )

            # Add constraints on observability windows for each field
            with status("adding field of regard constraints"):
                for mission in missions.keys():
                    for time_visit_vars, exptime, intervals in zip(
                        time_field_visit_vars[mission],
                        np.tile(visit_exptime_min_s, (n_fields[mission], 1)),
                        observable_intervals[mission],
                    ):
                        assert len(intervals) > 0
                        begin, end = intervals.T
                        if len(intervals) == 1:
                            model.add_constraints_(
                                time_visit_vars - begin - 0.5 * exptime >= 0
                            )
                            model.add_constraints_(
                                time_visit_vars - end + 0.5 * exptime <= 0
                            )
                        else:
                            visit_interval_vars = model.binary_vars(
                                (visits, len(intervals))
                            )
                            for interval_vars in visit_interval_vars:
                                model.add_constraint_(
                                    model.sum_vars_all_different(interval_vars) >= 1
                                )
                            exptime_per_visit = exptime[..., np.newaxis]
                            model.add_indicators(
                                visit_interval_vars,
                                time_visit_vars[:, np.newaxis]
                                - begin
                                - 0.5 * exptime_per_visit
                                >= 0,
                            )
                            model.add_indicators(
                                visit_interval_vars,
                                time_visit_vars[:, np.newaxis]
                                - end
                                + 0.5 * exptime_per_visit
                                <= 0,
                            )

            # Two observations are separated by half of each of their exposure
            # times, so a pair drawn from consecutive visits is separated by the
            # mean of theirs. Both the cadence and the slew constraints below
            # measure that separation.
            mean_consecutive_exptime_s = 0.5 * (
                visit_exptime_min_s[:-1] + visit_exptime_min_s[1:]
            )

            with status("adding slew constraints"):
                for mission in missions.keys():
                    slew_i, slew_j = slew_indices[mission]
                # Zero or less unless both fields are observed, which relaxes
                # the constraint away for any pair that is not.
                    both_observed = field_vars[mission][slew_i] + field_vars[mission][slew_j] - 1
                    # Two observations also clear each other by the slew itself.
                    def _spacing(exptimes):
                        return (
                            slew_time_s[mission][np.newaxis, :] + exptimes[:, np.newaxis]
                        ) * both_observed[np.newaxis, :]

                    rhs = rhs_within = _spacing(visit_exptime_min_s)
                    rhs_after = _spacing(mean_consecutive_exptime_s)

                    p, q = full_indices(visits)
                    if not adaptive_exptime:
                        rhs = _spacing(
                            0.5 * (visit_exptime_min_s[p] + visit_exptime_min_s[q])
                        )
                    model.add_constraints_(
                        model.abs(
                            time_field_visit_vars[mission][slew_i, p[:, np.newaxis]]
                            - time_field_visit_vars[mission][slew_j, q[:, np.newaxis]]
                        )
                        >= rhs
                    )

            with status("adding coverage constraints"):
                coverage_rhs = []
                for p in range(n_pixels_global):    
                    terms = [                       
                        field_vars[mission][f_idx]  
                        for mission in missions     
                        for f_idx in pixels_to_fields_map_global[mission][p]
                    ]
                    coverage_rhs.append(
                        model.sum_vars_all_different(terms) if terms else 0
                    )
                model.add_constraints_(pixel_vars <= coverage_rhs)
            # pixel_vars[p] = 1 iff at least one field of one mission observes it (pixel p)
            # pixel_vars[p] is a binary var so multiple missions can have filed_vars[m][f] = 1 on the same pix
            # but its probab will be counted only once -> if only pixel was observed once problem bc
            # diff telescopes have diff tiles sont we need overlapping of some tiles to have a maximum coverage

            with status("adding cuts"):
                for mission in missions.keys():
                    model.add_user_cut_constraint(
                        model.sum_vars_all_different(field_vars[mission])
                        <= (deadline - delay).to_value(u.s) / visit_exptime_min_s.sum()
                    )

            with status("adding objective function"):
                model.maximize(
                    model.scal_prod_vars_all_different(
                        pixel_vars, skymap_flat_global["PROB"]
                    )
                )

        if has_model := (
            model.number_of_constraints + model.objective_expr.number_of_terms() > 0
        ):
            with status("solving MILP model"):
                if write_progress is not None:
                    model.add_progress_listener(recorder := ProgressDataRecorder())

                if write_model is not None: # allows to create a directory for the model file if it doesn't exist
                    Path(write_model.name).parent.mkdir(parents=True, exist_ok=True)
                    model.to_stream(write_model)

                solution = model.solve()

                if solution is None:
                    print(
                        "No feasible solution was found before the solver stopped:",
                        model.solve_details.status,
                    )
                else:
                    print("objective:", solution.get_objective_value())
                    for m in missions:
                        vals = np.asarray(solution.get_values(field_vars[m]))
                        print(
                            f"{m}: {int(vals.sum())} fields selected "
                            f"out of {n_fields[m]}"
                        )
        else:
            solution = None

        with status("writing results"):
            if write_progress is not None:
                QTable(
                    rows=recorder.recorded if has_model else (),
                    names=ProgressData._fields,
                    dtype=[int, bool, float, float, float, int, int, int, float, float],
                ).write(write_progress, format="ascii.ecsv", overwrite=True)


            meta_common = {
                "deadline": deadline,         
                "delay": delay,
                "skygrid": skygrid,
                "nside": nside,
                "max_fields": max_fields,
                "time_step": time_step,
                "skymap": skymap.name,
                "event_time": event_time.isot,
                "visits": visits,            
                "exptime_min": exptime_min,
                "exptime_max": exptime_max,
                "absmag_mean": absmag_mean,
                "absmag_stdev": absmag_stdev,
                "appmag_dist": appmag_dist,
                "snr": snr,
                "cutoff": cutoff,
            }
            objective_value = (
                solution.get_objective_value() if solution is not None else 0.0
            )
            tables = {}
            for mission_name, mission_obj in missions.items():
                tables[mission_name] = build_mission_table(
                    mission_name=mission_name,
                    mission_obj=mission_obj,
                    field_vars=field_vars,
                    time_field_visit_vars=time_field_visit_vars,
                    n_fields=n_fields,
                    target_coords=target_coords,
                    rolls=rolls,
                    field_ids=field_ids,
                    visit_bandpasses=visit_bandpasses,
                    solution=solution,
                    visits=visits,      
                    visit_exptime_min_s=visit_exptime_min_s,
                    obstimes=obstimes,
                    deadline=deadline,                 
                    delay=delay,                       
                    meta_common=meta_common,
                    objective_value=objective_value,
                    model=model,
                    has_model=has_model,
                )

            for mission_name, table in tables.items():
                table.write(f"{schedule}_{mission_name}.ecsv", format="ascii.ecsv", overwrite=True)