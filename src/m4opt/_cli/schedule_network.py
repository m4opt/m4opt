import heapq
import inspect
import shlex
import sys
import tomllib
from itertools import pairwise  # noqa: F401
from pathlib import Path
from types import SimpleNamespace, UnionType
from typing import Annotated, Literal, Union, get_args, get_origin

import healpy as hp
import numpy as np
import synphot  # noqa: F401
import typer
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord  # noqa: F401
from astropy.table import QTable, vstack
from astropy.time import Time
from astropy_healpix import HEALPix
from click import UsageError
from docplex.mp.progress import ProgressData, ProgressDataRecorder
from ligo.skymap import distance  # noqa: F401
from ligo.skymap.bayestar import rasterize
from ligo.skymap.io import read_sky_map
from pydantic import TypeAdapter
from scipy import stats  # noqa: F401

from m4opt import __version__
from m4opt import networks as net
from m4opt._cli.core import app
from m4opt.dynamics import nominal_roll
from m4opt.fov import footprint_healpix
from m4opt.milp import Model
from m4opt.observer import EarthFixedObserverLocation
from m4opt.synphot import TabularScaleFactor, observing  # noqa: F401
from m4opt.synphot.extinction import DustExtinction  # noqa: F401
from m4opt.utils.console import progress, status
from m4opt.utils.numpy import clump_nonzero_inclusive, full_indices

from ._types import PhysicalQuantity
from .schedule import (
    invert_footprints,
    invert_footprints_to_regions,  # noqa: F401
    prepare_piecewise_breakpoints,  # noqa: F401
)

"""
Implementation of a multiple missions scheduler using a network class.
Work in progress
"""


# Helper class/functions to have different configurations for each mission of the netwrok
class PerMission:
    """Helper class: can overwrite an option via a config file."""

    def __init__(self, parser=None):
        self.parser = parser  # None -> pydantic validation after annotation


def _per_mission_spec(func):
    """{param: (type, parser)} for the PerMission marked parameters."""
    spec = {}
    for name, p in inspect.signature(func).parameters.items():
        if get_origin(p.annotation) is Annotated:
            base, *meta = get_args(p.annotation)
            marker = next((m for m in meta if isinstance(m, PerMission)), None)
            if marker:
                spec[name] = (base, marker.parser)
    return spec


def _is_list_type(tp):
    if get_origin(tp) in (Union, UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
    else:
        args = [tp]
    return len(args) == 1 and get_origin(args[0]) is list


def _parse(name, tp, parser, value):
    try:
        if parser is None:
            if _is_list_type(tp) and not isinstance(value, list):
                value = [value]  # cheat for the bandpass list for a unique bandpass
            parser = TypeAdapter(tp).validate_python
        return parser(value)
    except Exception as e:
        raise UsageError(
            f"Invalid value for '{name}' in config: {value!r} ({e})"
        ) from None


def load_config(path, network, spec):
    """Return {section_lower: {param: parsed value}}."""
    if path is None:
        return {}
    path = Path(path)
    with open(path, "rb") as f:
        if path.suffix in (".yml", ".yaml"):
            import yaml

            raw = yaml.safe_load(f) or {}
        elif path.suffix == ".toml":
            raw = tomllib.load(f)
        else:
            raise UsageError("Config file must be .yml, .yaml or .toml")

    names = {m.name.lower() for m in network}
    if "default" in names:
        raise UsageError("'default' is reserved and can't be a mission name.")

    cfg = {}
    for section, values in raw.items():
        key = section.lower()
        if key != "default" and key not in names:
            raise UsageError(
                f"Section '{section}' is neither 'default' nor a mission of network "
                f"'{network.name}' ({', '.join(sorted(names))})."
            )
        unknown = set(values) - spec.keys()
        if unknown:
            raise UsageError(
                f"[{section}] unknown parameter(s): {sorted(unknown)}. "
                f"Choose from: {', '.join(spec)}."
            )
        cfg[key] = {n: _parse(n, *spec[n], v) for n, v in values.items()}
    return cfg


def resolve_params(network, cfg, spec, cli_values, visits):
    """Priority : mission specific in a section > section 'default' > CLI arg."""
    default_cfg = cfg.get("default", {})
    params = {}
    for m in network:
        mine = cfg.get(m.name.lower(), {})
        d = {n: mine.get(n, default_cfg.get(n, cli_values[n])) for n in spec}
        ns = SimpleNamespace(**d)
        ns.visit_bandpasses, ns.visit_exptime_min_s = prepare_visit_configuration(
            visits, d["bandpass"], d["exptime_min"]
        )
        params[m.name] = ns
    return params


LARGE_EXPTIME = 1e10


def remap_footprints(footprints, pixel_map):
    return [pixel_map[footprint] for footprint in footprints]


def prepare_visit_configuration(
    visits, bandpass, exptime_min
):  # need modification because for now filter change is not really supported but diff filter per mission should be ok
    if visits < 1:
        raise UsageError("The number of visits must be at least one.")
    if not exptime_min:
        raise UsageError("Specify at least one minimum exposure time.")

    visit_bandpasses = [
        bandpass[i % len(bandpass)] if bandpass else None for i in range(visits)
    ]
    visit_exptime_min_s = u.Quantity(
        [exptime_min[i % len(exptime_min)] for i in range(visits)]
    ).to_value(u.s)
    return visit_bandpasses, visit_exptime_min_s


def build_mission_table(
    mission_name,
    mission_obj,  # missions[mission_name]
    *,
    field_vars,  # dict
    time_field_visit_vars,  # dict
    n_fields,  # dict
    target_coords,
    rolls,
    field_ids,  # dicts
    visit_bandpasses,  # dict
    solution,
    visits,
    skygrid,
    max_fields,
    exptime_min,
    exptime_max,
    visit_exptime_min_s,
    obstimes,
    deadline,
    delay,
    meta_common,  # common val for all missions
    objective_value,
    model,
    has_model,
):
    nf = n_fields[mission_name]

    # per mission value extraction
    if solution is None:
        field_values = np.zeros(nf, dtype=bool)
        time_field_visit_values = np.empty((nf, visits))
        exptime_field_values = np.empty((nf, visits))
    else:
        field_values = solution.get_values(field_vars[mission_name]) >= 0.5
        time_field_visit_values = solution.get_values(
            time_field_visit_vars[mission_name]
        )
        exptime_field_values = np.tile(visit_exptime_min_s, (nf, 1))

    # per mission table
    idx = np.flatnonzero(field_values)
    n_obs = idx.size * visits
    print(
        "visit_exptime_min_s:",
        type(visit_exptime_min_s),
        getattr(visit_exptime_min_s, "shape", None),
        visit_exptime_min_s,
    )
    table = QTable(
        {
            "action": np.full(n_obs, "observe"),
            "start_time": obstimes[0]
            + (
                time_field_visit_values[field_values]
                - 0.5 * exptime_field_values[field_values]
            ).ravel()
            * u.s,
            "duration": exptime_field_values[field_values].ravel() * u.s,
            "target_coord": target_coords[mission_name][
                np.tile(idx[:, np.newaxis], visits)
            ].ravel(),
            "roll": rolls[mission_name][np.tile(idx[:, np.newaxis], visits)].ravel(),
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
        },  # identiques
    )
    table.sort("start_time")

    # mission's intrinsic observer (not common)
    table.add_column(
        mission_obj.observer_location(table["start_time"]),
        index=3,
        name="observer_location",
    )
    table["observer_location"].info.description = "Position of the spacecraft"

    # slew
    if len(table) > 0:
        nrows = len(table) - 1
        slew_duration = mission_obj.slew.time(
            table["target_coord"][:-1],
            table["target_coord"][1:],
            table["roll"][:-1],
            table["roll"][1:],
        )
        changed = table["bandpass"][:-1] != table["bandpass"][1:]
        slew_duration[changed] = np.maximum(
            slew_duration[changed], mission_obj.filter_exchange_time
        )
        slew_table = QTable(
            {
                "action": np.full(nrows, "slew"),
                "start_time": (table["start_time"] + table["duration"])[:-1],
                "duration": slew_duration,
                "bandpass": np.full(nrows, ""),
            }
        )
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
            deadline - delay - total_time_by_action["duration"].sum()
        ).to(u.s)
    else:
        table.meta["total_time"] = {}

    table.meta.update(
        {
            "command": shlex.join(sys.argv),
            "version": __version__,
            "args": {
                **meta_common,
                "mission": mission_obj.name,
                "bandpass": visit_bandpasses,
                "deadline": deadline,
                "delay": delay,
                "skygrid": skygrid,
                "max_fields": max_fields,
                "exptime_min": exptime_min,
                "exptime_max": exptime_max,
            },
            "objective_value": objective_value,
            "best_bound": model.best_bound if has_model else 0,
            "solution_status": (
                model.solve_details.status
                if has_model
                else "infeasible, no observable fields or pixels"
            ),
            "solution_time": (model.solve_details.time if has_model else 0) * u.s,
        }
    )
    return table


@app.command()
@progress()
def schedule_network(
    skymap: Annotated[
        typer.FileBinaryRead,
        typer.Argument(help="Sky map filename", metavar="INPUT.multiorder.fits"),
    ],
    schedule: Annotated[
        str,
        typer.Argument(
            help="Output filename for generated schedule, attention, don't provide the extension, the scheduler will produce several out files (1 per mission)",
            metavar="SCHEDULE.ecsv",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="YAML/TOML file with per-mission parameters (max_fields, bandpass, exptime_min). "
            "Takes precedence over the CLI options",
        ),
    ] = None,
    exptime_min: Annotated[
        list[PhysicalQuantity[u.physical.time]],
        PerMission(),
        typer.Option(
            help="Minimum exposure time for each observation. Repeat the "
            "option to give each bandpass its own exposure time, in the same "
            "order as --bandpass; a single value applies to every bandpass",
        ),
    ] = (30 * u.s,),
    skygrid: Annotated[
        str | None,
        PerMission(),
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
        PerMission(),
        typer.Option(
            help="Delay from time of event until the start of observations",
        ),
    ] = 0 * u.day,
    deadline: Annotated[
        PhysicalQuantity[u.physical.time],
        PerMission(),
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
        PerMission(),
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
        PerMission(),
        typer.Option(
            help="Name of detector bandpass. Repeat the option to cycle through "
            "several bandpasses on successive visits; for example, "
            "--bandpass g --bandpass r observes each field in g and then in r. "
            "Visits are grouped into contiguous blocks of a single bandpass so "
            "that the filter is exchanged only between blocks."
        ),
    ] = None,  # must be changed to have per facility filter !
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 1,
    cadence: Annotated[
        PhysicalQuantity[u.physical.time],
        typer.Option(help="Minimum time separation between visits"),
    ] = 30 * u.min,
    nside: Annotated[int, typer.Option(help="HEALPix resolution")] = 512,
    max_fields: Annotated[
        int,
        PerMission(),
        typer.Option(
            min=1,
            help="Consider only this many of the most probable fields. Raising "
            "it grows the problem roughly quadratically",
        ),
    ] = 50,
    preselection_method: Annotated[
        Literal["pixel", "tile"],
        typer.Option(
            help="Method to use for the preselection of observables tiles to schedule:"
            'Possible options are "pixel" or "tile"'
            "`pixel`: Preselection is done jointly and avoid recounting a pixel's probability more than once - more adapted to a fixed exposure time. (can lead to small gaps in the coverage)"
            "`tile`: Preselection is done by selectionning the most probable tile for each telescopes independantly - more adapted to an adaptive exposure time."
        ),
    ] = "pixel",
    overlap_weight: Annotated[
        float,
        typer.Option(
            min=0,
            max=1,
            help="Weight for the overlap term in the objective function. 0=no overlap, 1=recount each pixel when preselecting the most probable fields",
        ),
    ] = 1,
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
    network: Annotated[
        net.Network,
        typer.Option(
            "--network",
            help="Network of telescopes to use for the scheduler.",
            show_default="ground",
        ),
    ] = net.ground,
    write_candidates: Annotated[
        str | None,
        typer.Option(
            help="Save observable candidate fields and their HEALPix footprints for each mission to an NPZ file. The input is a suffix, the saved files will be named as <suffix>_<mission>.npz",
            metavar="<suffix>_<mission>.npz",
        ),
    ] = None,
):
    r"""
    Generate an observing plan for a (GW) skymap.

    This is a work in progress, major changes are expected.
    For now, only supports a fixed exposure time.

    Planned changes:
    - Implement the other two modes of `m4opt schedule` (adaptative exposure time)

    # m4opt schedule-network

    ### Current state: only a mono-filter, fixed exposure time version (adaptation of the [problem 1](https://m4opt.readthedocs.io/en/latest/design/milp.html#problem-1-fixed-exposure-time) from the documentation)

    ### Different preselection:
    The network scheduler can be used with two different preselection methods:
    - `pixel`: the preselection of fields is now made jointly by computing the marginal probability gain (+ an overlap weight targeting the tiles edges) of a tile using `heapq`. The overlap weight is a variable between 0 and 1 allowing more or less overlap between the tiles. With 0 overlap, no tiles will ever superpose (unlike in the classic scheduler).
    - `tile`: the preselection of fields is the same as in `m4opt schedule` with fields being preselected independantly for each telescope of the network (will lead to superposition of preselected tiles). For a given number of tiles, will cover a smaller surface than the pixel method.

    ### Constraints:
    - FoV, slew, cuts exactly the same → **mission specific constraints grows linearly with the number of facilities**
    - Joint coverage constraints: `pixel_vars` is now shared across all facilities, concatenate all individual missions' `good pixels`.

    ### Configuration:
    A `.yaml/yml/toml`configuration file can be passed as an option to the CLI to specify per missions parameters. The parameters declared in the configuration file will override the CLI options.
    Supported parameters are:
    - `exptime_min`
    - `skygrid`
    - `delay`
    - `deadline`
    - `exptime_max`
    - `bandpass`
    - `max_fields`

    ### Objective
    Same implementation (so $\max \sum_i \rho_i \, p_i$) but with

    $$p_i \le \sum_f \sum_{j \in J_i^{(f)}} m_{f,j} \quad \forall i$$

    With:
    - $\sum_f$ a sum over the different facilities (missions) considered
    - $m_{f,j}$ is the $j$-th field of facility $f$
    - $J_i^{(f)}$ is the set of indices of fields that contain the pixel $i$ for facility $f$

    **Note:** The only shared var important to the MILP problem is the pixels covered. Hence, pixel probability is only counted once even if multiple telescopes observe it.
    """
    # parse config
    spec = _per_mission_spec(schedule_network)
    cfg = load_config(config, network, spec)
    loc = locals()
    cli_values = {n: loc[n] for n in spec}
    params = resolve_params(network, cfg, spec, cli_values, visits)

    with status(f"Using network: {network.name}"):
        adaptive_exptime = False  # force constant exp time for now

    if adaptive_exptime and bandpass is not None and len(bandpass) > 1:
        raise NotImplementedError(
            "A variable exposure time is not supported with more than one bandpass."
        )
    # filter_changes = [lhs != rhs for lhs, rhs in pairwise(visit_bandpasses)] # not usable for now

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
        observer_locations = {}
        obstimes = {}
        for mission in network:
            obstimes[mission.name] = event_time + np.arange(
                params[mission.name].delay,
                params[mission.name].deadline + time_step,
                time_step,
                like=time_step,
            )
            observer_locations[mission.name] = mission.observer_location(
                obstimes[mission.name]
            )

    with status("evaluating field of regard"):
        target_coords = {}
        field_ids = {}
        observable_intervals = {}
        for mission in network:
            if not isinstance(mission.skygrid, dict):
                target_coords[mission.name] = mission.skygrid
            elif params[mission.name].skygrid in mission.skygrid:
                target_coords[mission.name] = mission.skygrid[
                    params[mission.name].skygrid
                ]
            else:
                raise UsageError(
                    f"skygrid '{params[mission.name].skygrid}' not found. Options: {', '.join(map(str, mission.skygrid))}"
                )

            # The row of the grid is the mission's own name for the field. A
            # mission that numbers its fields leaves gaps, masked out of the grid
            # and dropped here so that everything below is dense.
            keep = ~target_coords[mission.name].mask
            field_ids[mission.name] = np.arange(len(target_coords[mission.name]))[keep]
            target_coords[mission.name] = target_coords[mission.name].unmasked[keep]
            # FIXME: https://github.com/astropy/astropy/issues/17030
            target_coords[mission.name] = SkyCoord(
                target_coords[mission.name].ra, target_coords[mission.name].dec
            )
            cadence_s = cadence.to_value(
                u.s
            )  # now only fix cadence so no problem leaving it like this
            obs = obstimes[mission.name]
            obstimes_s = (obs - obs[0]).to_value(u.s)
            observable_intervals[mission.name] = np.asarray(
                [
                    obstimes_s[intervals]
                    for intervals in clump_nonzero_inclusive(
                        mission.constraints(
                            observer_locations[mission.name],
                            target_coords[mission.name][:, np.newaxis],
                            obs,
                        )
                    )
                ],
                dtype=object,
            )

            # Keep only intervals that are at least as long as the exposure time.
            exptime_min_s = params[mission.name].visit_exptime_min_s.min()
            observable_intervals[mission.name] = np.asarray(
                [
                    intervals[intervals[:, 1] - intervals[:, 0] >= exptime_min_s]
                    for intervals in observable_intervals[mission.name]
                ],
                dtype=object,
            )

            # Discard fields that are not observable.
            good = np.asarray(
                [len(intervals) > 0 for intervals in observable_intervals[mission.name]]
            )
            observable_intervals[mission.name] = observable_intervals[mission.name][
                good
            ]
            target_coords[mission.name] = target_coords[mission.name][good]
            field_ids[mission.name] = field_ids[mission.name][good]

    with status("calculating footprints"):
        rolls = {}
        global_good_pixels = {}  # try to have a unique pixel var for multiple mission
        footprints = {}
        n_fields = {}
        n_pixels = {}
        skymap_flat_full = skymap_flat
        all_footprints = {}
        for mission in network:
            if isinstance(mission.observer_location, EarthFixedObserverLocation):
                rolls[mission.name] = np.zeros(len(target_coords[mission.name])) * u.deg
            else:
                # Compute nominal roll angles for optimal solar power.
                # The nominal roll angle varies as a function of sky position and time.
                # We compute it for the start of the schedule because we assume that it
                # does not change much over the duration.
                rolls[mission.name] = nominal_roll(
                    observer_locations[mission.name][0],
                    target_coords[mission.name],
                    event_time,
                )
            footprints[mission.name] = footprint_healpix(
                hpx, mission.fov, target_coords[mission.name], rolls[mission.name]
            )
            all_footprints[mission.name] = footprints[mission.name]

        if preselection_method == "tile":
            # use the same "tile" approach as the default `schedule`
            for mission in network:
                n_fields[mission.name] = params[mission.name].max_fields
                if len(target_coords[mission.name]) > n_fields[mission.name]:
                    good = np.argpartition(
                        [
                            -skymap_flat_full[footprint]["PROB"].sum()
                            for footprint in footprints[mission.name]
                        ],
                        n_fields[mission.name],
                    )[: n_fields[mission.name]]
                    target_coords[mission.name] = target_coords[mission.name][good]
                    rolls[mission.name] = rolls[mission.name][good]
                    footprints[mission.name] = footprints[mission.name][good]
                    observable_intervals[mission.name] = observable_intervals[
                        mission.name
                    ][good]
                    field_ids[mission.name] = field_ids[mission.name][good]
                else:
                    n_fields[mission.name] = len(target_coords[mission.name])
        else:
            pix_prob = np.asarray(skymap_flat_full["PROB"])
            counts = np.zeros(len(pix_prob), dtype=np.int32)
            mask = np.zeros(len(pix_prob), dtype=bool)  # buffer used for tile erosion
            edgeness = np.zeros(len(pix_prob), dtype=np.float32)

            selected = {mission.name: [] for mission in network}

            def tile_edgeness(fp, rim_frac):
                """Trim the edges of a tile to allow for a smart(er) overlap weight"""
                fp = np.asarray(fp)
                rim_px = max(
                    1, int(rim_frac * np.sqrt(len(fp)))
                )  # scale acordingly to the used mission & skymap
                nb = hp.get_all_neighbours(
                    nside, fp, nest=True
                )  # focus on neighboring pixels: try to fill the gaps
                valid = nb >= 0
                mask[fp] = True
                depth = np.full(len(fp), rim_px, dtype=np.int16)
                alive = np.ones(len(fp), dtype=bool)
                for k in range(rim_px):
                    inside = mask[np.where(valid, nb, 0)] & valid
                    edge = alive & ~inside.all(axis=0)  # a un voisin hors tuile
                    if not edge.any():
                        break
                    depth[edge] = k
                    alive &= ~edge
                    mask[fp[edge]] = False  # on rogne une couche
                mask[fp] = False
                return np.clip(1 - depth / rim_px, 0, 1).astype(np.float32)

            def marginal_gain(fp):
                fp = np.asarray(fp)
                p, old = pix_prob[fp], counts[fp] > 0
                return (
                    p[~old].sum() + overlap_weight * (p[old] * edgeness[fp[old]]).sum()
                )

            heap = []
            for mission in network:
                for i, fp in enumerate(footprints[mission.name]):
                    heapq.heappush(heap, (-marginal_gain(fp), mission.name, i))

            n_covered_version = 0
            gain_version = {}
            full = set()

            while heap and len(full) < len(network):
                neg_gain, name, i = heapq.heappop(heap)
                if name in full:
                    continue
                if gain_version.get((name, i), -1) != n_covered_version:
                    gain_version[(name, i)] = n_covered_version
                    heapq.heappush(heap, (-marginal_gain(footprints[name][i]), name, i))
                    continue
                if -neg_gain <= 0:
                    break
                selected[name].append(i)
                fp_sel = np.asarray(footprints[name][i])
                e = tile_edgeness(fp_sel, rim_frac=0.15)
                edgeness[fp_sel] = np.where(
                    counts[fp_sel] == 0, e, np.minimum(edgeness[fp_sel], e)
                )
                counts[fp_sel] += 1
                n_covered_version += 1
                if len(selected[name]) >= params[name].max_fields:
                    full.add(name)
            print(
                f"Found best fields after {n_covered_version} iterations, covering {np.count_nonzero(counts)} pixels"
            )
            for mission in network:
                good = np.array(sorted(selected[mission.name]), dtype=int)
                n_fields[mission.name] = len(good)
                target_coords[mission.name] = target_coords[mission.name][good]
                rolls[mission.name] = rolls[mission.name][good]
                footprints[mission.name] = footprints[mission.name][good]
                observable_intervals[mission.name] = observable_intervals[mission.name][
                    good
                ]
                field_ids[mission.name] = field_ids[mission.name][good]
        for mission in network:
            # save the candidates footprint for a later use
            if write_candidates is not None:
                write_candidates_mission = f"{write_candidates}_{mission.name}.npz"
                candidate_footprints = [
                    np.asarray(footprint, dtype=np.intp)
                    for footprint in footprints[mission.name]
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
                    field_id=field_ids[mission.name],
                    ra_deg=target_coords[mission.name].ra.to_value(u.deg),
                    dec_deg=target_coords[mission.name].dec.to_value(u.deg),
                    roll_deg=rolls[mission.name].to_value(u.deg),
                    probability=candidate_probability,
                    footprint_pixels=candidate_pixels,
                    footprint_offsets=footprint_offsets,
                    nside=nside,
                    order="nested",
                )

            # Throw away pixels that are not contained in any fields.
            good_pixels = (
                np.unique(np.concatenate(footprints[mission.name]))
                if len(footprints[mission.name]) > 0
                else np.asarray([], dtype=np.intp)
            )
            global_good_pixels[mission.name] = good_pixels
            imap = np.empty(len(skymap_flat_full), dtype=np.intp)
            imap[good_pixels] = np.arange(len(good_pixels))
            footprints[mission.name] = remap_footprints(footprints[mission.name], imap)
            n_pixels[mission.name] = len(good_pixels)

            # pixels_to_fields_map[mission.name] = invert_footprints(footprints[mission.name], n_pixels[mission.name]) -> useless ?

    with status("calculating slew times"):
        slew_time_s = {}
        slew_indices = {}
        for mission in network:
            slew_i, slew_j = np.triu_indices(n_fields[mission.name], 1)
            slew_indices[mission.name] = (slew_i, slew_j)
            slew_time_s[mission.name] = mission.slew.time(
                target_coords[mission.name][slew_i],
                target_coords[mission.name][slew_j],
                rolls[mission.name][slew_i],
                rolls[mission.name][slew_j],
            ).to_value(u.s)

    # construct global pixel var
    all_pixels_global = np.unique(
        np.concatenate([global_good_pixels[m.name] for m in network])
    )
    if len(all_pixels_global) == 0:
        raise ValueError("No observable pixels")
    n_pixels_global = len(all_pixels_global)
    assert n_pixels_global >= max(n_pixels[m.name] for m in network), (
        "n_pixels_global < max local : bug dans l'union"
    )  # AI assert as a failsafe but should never appear

    # remap HEALPIX index to the all pixel var
    global_map = np.full(len(skymap_flat_full), -1, dtype=np.intp)
    global_map[all_pixels_global] = np.arange(n_pixels_global)

    skymap_flat_global = skymap_flat_full[all_pixels_global]

    # Remap each mission's footprint to the global footprint space
    footprints_global = {}
    pixels_to_fields_map_global = {}
    for mission in network:
        local_to_global = global_map[global_good_pixels[mission.name]]
        footprints_global[mission.name] = remap_footprints(
            footprints[mission.name], local_to_global
        )
        assert all((f >= 0).all() for f in footprints_global[mission.name]), (
            f"missing pixel for the union of mission {mission}"
        )
        pixels_to_fields_map_global[mission.name] = invert_footprints(
            footprints_global[mission.name], n_pixels_global
        )

    print(
        "skymap_full PROB:",
        float(skymap_flat_full["PROB"].min()),
        float(skymap_flat_full["PROB"].max()),
        float(skymap_flat_full["PROB"].mean()),
        float(skymap_flat_full["PROB"].sum()),
    )  # ← sum ≈ 1 ?
    print("skymap_full n_pixels:", len(skymap_flat_full))
    print(
        "all_pixels_global min/max:", all_pixels_global.min(), all_pixels_global.max()
    )
    print(
        "frac PROB covered:",
        float(
            skymap_flat_full["PROB"][all_pixels_global].sum()
            / skymap_flat_full["PROB"].sum()
        ),
    )

    nside_skymap = hp.npix2nside(len(skymap_flat_full))
    ipix_peak = int(np.argmax(skymap_flat_full["PROB"]))
    ra_peak, dec_peak = hp.pix2ang(nside_skymap, ipix_peak, lonlat=True)
    print(
        f"skymap peak: RA={ra_peak:.2f} Dec={dec_peak:.2f} PROB={skymap_flat_full['PROB'][ipix_peak]:.3e}"
    )

    for m in network:
        tc = target_coords[m.name]
        if len(tc) == 0:
            print(
                f"{m.name}: n_obs=0, no fields are observable in the scheduling window"
            )
            continue
        sep = tc.separation(SkyCoord(ra_peak * u.deg, dec_peak * u.deg)).deg
        i_best = int(np.argmin(sep))
        print(
            f"{m.name}: n_obs={len(tc)}, best field to peak = {sep[i_best]:.2f} deg, "
            f"field RA/Dec = ({tc[i_best].ra.deg:.2f}, {tc[i_best].dec.deg:.2f})"
        )

    # do the MILP model
    with Model(
        timelimit=timelimit, jobs=jobs, memory=memory, lowercutoff=cutoff
    ) as model:
        with status("assembling MILP model"):
            pixel_vars = model.binary_vars(n_pixels_global)

            field_vars = {}
            time_field_visit_vars = {}
            for mission in network:
                field_vars[mission.name] = model.binary_vars(n_fields[mission.name])
                time_field_visit_vars[mission.name] = model.continuous_vars(
                    (n_fields[mission.name], visits),
                )

            # Add constraints on observability windows for each field
            with status("adding field of regard constraints"):
                for mission in network:
                    for time_visit_vars, exptime, intervals in zip(
                        time_field_visit_vars[mission.name],
                        np.tile(
                            params[mission.name].visit_exptime_min_s,
                            (n_fields[mission.name], 1),
                        ),
                        observable_intervals[mission.name],
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
                    params[mission.name].mean_consecutive_exptime_s = 0.5 * (
                        params[mission.name].visit_exptime_min_s[:-1]
                        + params[mission.name].visit_exptime_min_s[1:]
                    )

            with status("adding slew constraints"):
                for mission in network:
                    slew_i, slew_j = slew_indices[mission.name]
                    # Zero or less unless both fields are observed, which relaxes
                    # the constraint away for any pair that is not.
                    both_observed = (
                        field_vars[mission.name][slew_i]
                        + field_vars[mission.name][slew_j]
                        - 1
                    )

                    # Two observations also clear each other by the slew itself.
                    def _spacing(exptimes):
                        return (
                            slew_time_s[mission.name][np.newaxis, :]
                            + exptimes[:, np.newaxis]
                        ) * both_observed[np.newaxis, :]

                    rhs = rhs_within = _spacing(
                        params[mission.name].visit_exptime_min_s
                    )
                    rhs_after = _spacing(
                        params[mission.name].mean_consecutive_exptime_s
                    )

                    p, q = full_indices(visits)
                    if not adaptive_exptime:
                        rhs = _spacing(
                            0.5
                            * (
                                params[mission.name].visit_exptime_min_s[p]
                                + params[mission.name].visit_exptime_min_s[q]
                            )
                        )
                    model.add_constraints_(
                        model.abs(
                            time_field_visit_vars[mission.name][
                                slew_i, p[:, np.newaxis]
                            ]
                            - time_field_visit_vars[mission.name][
                                slew_j, q[:, np.newaxis]
                            ]
                        )
                        >= rhs
                    )

            with status("adding coverage constraints"):
                coverage_rhs = []
                for p in range(n_pixels_global):
                    terms = [
                        field_vars[mission.name][f_idx]
                        for mission in network
                        for f_idx in pixels_to_fields_map_global[mission.name][p]
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
                for mission in network:
                    model.add_user_cut_constraint(
                        model.sum_vars_all_different(field_vars[mission.name])
                        <= (
                            params[mission.name].deadline - params[mission.name].delay
                        ).to_value(u.s)
                        / params[mission.name].visit_exptime_min_s.sum()
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

                if (
                    write_model is not None
                ):  # allows to create a directory for the model file if it doesn't exist
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
                    for m in network:
                        vals = np.asarray(solution.get_values(field_vars[m.name]))
                        print(
                            f"{m.name}: {int(vals.sum())} fields selected "
                            f"out of {n_fields[m.name]}"
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
                "nside": nside,
                "time_step": time_step,
                "skymap": skymap.name,
                "event_time": event_time.isot,
                "visits": visits,
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
            for mission in network:
                tables[mission.name] = build_mission_table(
                    mission_name=mission.name,
                    mission_obj=mission,
                    field_vars=field_vars,
                    time_field_visit_vars=time_field_visit_vars,
                    n_fields=n_fields,
                    target_coords=target_coords,
                    rolls=rolls,
                    field_ids=field_ids,
                    visit_bandpasses=params[mission.name].visit_bandpasses,
                    solution=solution,
                    visits=visits,
                    visit_exptime_min_s=params[mission.name].visit_exptime_min_s,
                    obstimes=obstimes[mission.name],
                    deadline=params[mission.name].deadline,
                    delay=params[mission.name].delay,
                    meta_common=meta_common,
                    objective_value=objective_value,
                    model=model,
                    has_model=has_model,
                    skygrid=params[mission.name].skygrid,
                    max_fields=params[mission.name].max_fields,
                    exptime_min=params[mission.name].exptime_min,
                    exptime_max=params[mission.name].exptime_max,
                )

            for mission_name, table in tables.items():
                table.write(
                    f"{schedule}_{mission_name}.ecsv",
                    format="ascii.ecsv",
                    overwrite=True,
                )
