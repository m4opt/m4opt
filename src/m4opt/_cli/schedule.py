import inspect
import json
import shlex
import sys
from itertools import pairwise
from typing import Annotated, get_args, get_origin, get_type_hints

import numpy as np
import synphot
import typer
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

from .. import __version__, constraints, missions
from ..dynamics import nominal_roll
from ..fov import footprint_healpix
from ..milp import Model
from ..observer import EarthFixedObserverLocation
from ..synphot import TabularScaleFactor, observing
from ..synphot.extinction import DustExtinction
from ..utils.console import progress, status
from ..utils.numpy import clump_nonzero_inclusive, full_indices
from .core import app


def invert_footprints(footprints, n_pixels):
    """
    Construct a map from pixel index to footprints containing those pixels.

    Examples
    --------
    >>> from m4opt._cli.schedule import invert_footprints
    >>> invert_footprints([[1, 2, 3], [0, 2, 3]], 4)
    [array([1]), array([0]), array([0, 1]), array([0, 1])]
    """
    pixels_to_fields_map = [[] for _ in range(n_pixels)]
    for i, js in enumerate(footprints):
        for j in js:
            pixels_to_fields_map[j].append(i)
    return [np.asarray(field_indices) for field_indices in pixels_to_fields_map]


def invert_footprints_to_regions(footprints, n_pixels):
    """
    Construct a map from pixel index to disjoint regions.

    Examples
    --------
    >>> from m4opt._cli.schedule import invert_footprints_to_regions
    >>> invert_footprints_to_regions([[1, 2, 3], [0, 2, 3]], 4)
    ([1, 0, 2, 2], [array([0]), array([1]), array([0, 1])])
    """
    pixels_to_fields_map = [
        tuple(field_indices)
        for field_indices in invert_footprints(footprints, n_pixels)
    ]
    region_to_fields_map = {
        footprint: i for i, footprint in enumerate(set(pixels_to_fields_map))
    }
    pixel_to_region_map = [
        region_to_fields_map[footprint] for footprint in pixels_to_fields_map
    ]
    region_to_fields_map = [
        np.asarray(fields, dtype=np.intp) for fields in region_to_fields_map
    ]
    return pixel_to_region_map, region_to_fields_map


LARGE_EXPTIME = 1e10


def prepare_piecewise_breakpoints(breakpoints):
    isinf_indices = np.flatnonzero(breakpoints[:, 1] >= LARGE_EXPTIME)
    if len(isinf_indices) > 0:
        breakpoints = breakpoints[: isinf_indices[0]]
    return [tuple(col.item() for col in row) for row in breakpoints]


# helper functions for parsing constraints from JSON on the command line
def _is_quantity_annotation(annotation):
    return (
        annotation is u.Quantity
        or get_origin(annotation) is u.Quantity
        or any(_is_quantity_annotation(arg) for arg in get_args(annotation))
    )


def _is_numeric_annotation(annotation):
    return annotation in (int, float) or any(
        _is_numeric_annotation(arg) for arg in get_args(annotation)
    )


def _is_skycoord_annotation(annotation):
    return annotation is SkyCoord or any(
        _is_skycoord_annotation(arg) for arg in get_args(annotation)
    )


def _parse_constraint_spec(spec):
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as error:
            raise UsageError(f"invalid constraint JSON: {error.msg}") from error

    if not isinstance(spec, dict) or set(spec) - {"type", "args", "kwargs"}:
        raise UsageError(
            "a constraint must be a JSON object with 'type', optional 'args', "
            "and optional 'kwargs' fields"
        )

    class_name = spec.get("type")
    if (
        not isinstance(class_name, str)
        or (constraint_class := getattr(constraints, class_name, None)) is None
        or not inspect.isclass(constraint_class)
        or not issubclass(constraint_class, constraints.Constraint)
        or inspect.isabstract(constraint_class)
    ):
        supported = [
            name
            for name in constraints.__all__
            if inspect.isclass(getattr(constraints, name))
            and issubclass(getattr(constraints, name), constraints.Constraint)
            and not inspect.isabstract(getattr(constraints, name))
        ]
        raise UsageError(
            f"unknown constraint type {class_name!r}; choose from {', '.join(supported)}"
        )

    args = spec.get("args", [])
    kwargs = spec.get("kwargs", {})
    if not isinstance(args, list) or not isinstance(kwargs, dict):
        raise UsageError("'args' must be a JSON array and 'kwargs' must be an object")

    constructor = inspect.signature(constraint_class)
    try:
        bound = constructor.bind(*args, **kwargs)
    except TypeError as error:
        raise UsageError(f"invalid arguments for {class_name}: {error}") from error

    hints = get_type_hints(constraint_class.__init__)
    for name, value in bound.arguments.items():
        parameter = constructor.parameters[name]
        annotation = hints.get(name, parameter.annotation)
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            bound.arguments[name] = tuple(
                _parse_constraint_value(item, annotation) for item in value
            )
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            bound.arguments[name] = {
                key: _parse_constraint_value(item, annotation)
                for key, item in value.items()
            }
        else:
            bound.arguments[name] = _parse_constraint_value(value, annotation)

    if constraint_class in (
        constraints.LogicalAndConstraint,
        constraints.LogicalOrConstraint,
    ) and not bound.arguments.get("operands"):
        raise UsageError(f"{class_name} requires at least one constraint operand")

    try:
        return constraint_class(*bound.args, **bound.kwargs)
    except (TypeError, ValueError) as error:
        raise UsageError(f"invalid arguments for {class_name}: {error}") from error


def _parse_constraint_value(value, annotation):
    if isinstance(value, dict) and "type" in value:
        return _parse_constraint_spec(value)
    if _is_skycoord_annotation(annotation):
        if isinstance(value, SkyCoord):
            return value
        try:
            return SkyCoord(**value) if isinstance(value, dict) else SkyCoord(value)
        except (TypeError, ValueError) as error:
            raise UsageError(
                f"invalid sky coordinate {value!r} in constraint"
            ) from error
    if _is_quantity_annotation(annotation) and not isinstance(value, str):
        if _is_numeric_annotation(annotation):
            return value
        raise UsageError("quantity constraint arguments must be strings with units")
    if isinstance(value, str) and (
        _is_quantity_annotation(annotation) or annotation is inspect.Parameter.empty
    ):
        try:
            return u.Quantity(value)
        except (TypeError, ValueError) as error:
            raise UsageError(f"invalid quantity {value!r} in constraint") from error
    return value


@app.command()
@progress()
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
    exptime_min: Annotated[
        list[u.Quantity[u.physical.time]],
        typer.Option(
            help="Minimum exposure time for each observation. Repeat the "
            "option to give each bandpass its own exposure time, in the same "
            "order as --bandpass; a single value applies to every bandpass",
        ),
    ],
    mission: Annotated[
        missions.Mission, typer.Option(show_default="uvex")
    ] = missions.uvex,
    additional_constraints: Annotated[
        list[str] | None,
        typer.Option(
            "--constraint",
            help="Supplementary built-in constraint as JSON. Repeat to add "
            "multiple constraints; each is ANDed with the mission defaults.",
            metavar="JSON",
        ),
    ] = None,
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
        u.Quantity,
        typer.Option(
            help="Delay from time of event until the start of observations",
        ),
    ] = 0 * u.day,
    deadline: Annotated[
        u.Quantity,
        typer.Option(
            help="Maximum time from event until the end of observations",
        ),
    ] = 1 * u.day,
    time_step: Annotated[
        u.Quantity,
        typer.Option(
            help="Time step for evaluating field of regard",
        ),
    ] = 1 * u.min,
    exptime_max: Annotated[
        u.Quantity,
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
    ] = None,
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 2,
    cadence: Annotated[
        u.Quantity,
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
    timelimit: Annotated[
        u.Quantity,
        typer.Option(
            help="Time limit for MILP solver",
            rich_help_panel="Solver Options",
        ),
    ] = 1e75 * u.s,
    memory: Annotated[
        u.Quantity,
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
):
    """
    Generate an observing plan for a GW sky map.

    \b
    The scheduler has three modes:

    \b
    1. Fixed exposure time. Every field has the same exposure time given by the
       --exptime-min option. This mode is selected if you omit the
       --absmag-mean option.

    \b
    2. Variable exposure time. Each field may have a different exposure time,
       adjusted for the posterior median distance along each line of sight.
       This mode is selected if you specify a value for the --absmag-mean
       option but also pass the --no-appmag-dist option.

    \b
    3. Variable exposure time with an absolute magnitude distribution. Each
       field may have a different exposure time, adjusted to optimize the
       detection probability given the posterior distance distribution and a
       Gaussian distribution of absolute magnitudes. This mode is selected if
       you specify the --absmag-mean option (and, optionally, the
       --absmag-stdev option).

    \b
    Repeating the --bandpass option makes successive visits cycle through the
    bandpasses; for example, --bandpass g --bandpass r observes every field in g
    and then every field in r. Visits are grouped into contiguous blocks of a single
    bandpass, so that every field is observed for the kth time before any field
    is observed for the (k+1)th, and the filter is exchanged once per block
    boundary however many fields are observed.

    \b
    Supplementary constraints are specified as JSON objects with a built-in
    constraint class name and optional constructor arguments. Quantity arguments
    are strings with units, for example:
    --constraint '{"type":"DeclinationConstraint","args":["-40 deg","90 deg"]}'.
    Repeat --constraint to add several constraints; all are combined with the
    mission constraints using logical AND. Nested constraint objects can be used
    as arguments to LogicalAndConstraint, LogicalOrConstraint, and
    LogicalNotConstraint. SkyCoord arguments may be coordinate strings such as
    "12h30m00s +45d00m00s" or objects such as
    {"ra":"12h30m00s","dec":"+45d00m00s"}. Supplementary constraints are added
    after the mission constraints.
    """
    adaptive_exptime = absmag_mean is not None
    mission_constraint = mission.constraints
    for constraint_spec in additional_constraints or ():
        mission_constraint &= _parse_constraint_spec(constraint_spec)

    # Successive visits cycle through the requested bandpasses, so that
    # --bandpass g --bandpass r over three visits gives g, r, g.
    visit_bandpasses = [
        bandpass[i % len(bandpass)] if bandpass else None for i in range(visits)
    ]
    visit_exptime_min_s = u.Quantity(
        [exptime_min[i % len(exptime_min)] for i in range(visits)]
    ).to_value(u.s)
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

    with status("propagating orbit"):
        obstimes = event_time + np.arange(
            delay, deadline + time_step, time_step, like=time_step
        )
        observer_locations = mission.observer_location(obstimes)

    with status("evaluating field of regard"):
        if not isinstance(mission.skygrid, dict):
            target_coords = mission.skygrid
        elif skygrid in mission.skygrid:
            target_coords = mission.skygrid[skygrid]
        else:
            raise UsageError(
                f"skygrid '{skygrid}' not found. Options: {', '.join(map(str, mission.skygrid.keys()))}"
            )

        # The row of the grid is the mission's own name for the field. A
        # mission that numbers its fields leaves gaps, masked out of the grid
        # and dropped here so that everything below is dense.
        keep = ~target_coords.mask
        field_ids = np.arange(len(target_coords))[keep]
        target_coords = target_coords.unmasked[keep]
        # FIXME: https://github.com/astropy/astropy/issues/17030
        target_coords = SkyCoord(target_coords.ra, target_coords.dec)
        cadence_s = cadence.to_value(u.s)
        obstimes_s = (obstimes - obstimes[0]).to_value(u.s)
        observable_intervals = np.asarray(
            [
                obstimes_s[intervals]
                for intervals in clump_nonzero_inclusive(
                    mission_constraint(
                        observer_locations,
                        target_coords[:, np.newaxis],
                        obstimes,
                    )
                )
            ],
            dtype=object,
        )

        # Keep only intervals that are at least as long as the exposure time.
        exptime_min_s = visit_exptime_min_s.min()
        observable_intervals = np.asarray(
            [
                intervals[intervals[:, 1] - intervals[:, 0] >= exptime_min_s]
                for intervals in observable_intervals
            ],
            dtype=object,
        )

        # Discard fields that are not observable.
        good = np.asarray([len(intervals) > 0 for intervals in observable_intervals])
        observable_intervals = observable_intervals[good]
        target_coords = target_coords[good]
        field_ids = field_ids[good]

    with status("calculating footprints"):
        if isinstance(mission.observer_location, EarthFixedObserverLocation):
            rolls = np.zeros(len(target_coords)) * u.deg
        else:
            # Compute nominal roll angles for optimal solar power.
            # The nominal roll angle varies as a function of sky position and time.
            # We compute it for the start of the schedule because we assume that it
            # does not change much over the duration.
            rolls = nominal_roll(observer_locations[0], target_coords, event_time)
        footprints = footprint_healpix(hpx, mission.fov, target_coords, rolls)

        # Consider only the most probable fields.
        n_fields = max_fields
        if len(target_coords) > n_fields:
            good = np.argpartition(
                [-skymap_flat[footprint]["PROB"].sum() for footprint in footprints],
                n_fields,
            )[:n_fields]
            target_coords = target_coords[good]
            rolls = rolls[good]
            footprints = footprints[good]
            observable_intervals = observable_intervals[good]
            field_ids = field_ids[good]
        else:
            n_fields = len(target_coords)

        # Throw away pixels that are not contained in any fields.
        good = (
            np.unique(np.concatenate(footprints))
            if len(footprints) > 0
            else np.asarray([], dtype=np.intp)
        )
        imap = np.empty(len(skymap_flat), dtype=np.intp)
        imap[good] = np.arange(len(good))
        skymap_flat = skymap_flat[good]
        footprints = np.asarray(
            [imap[footprint] for footprint in footprints], dtype=object
        )
        n_pixels = len(skymap_flat)

        if adaptive_exptime:
            pixel_to_region_map, region_to_fields_map = invert_footprints_to_regions(
                footprints, n_pixels
            )
            n_regions = len(region_to_fields_map)
        else:
            pixels_to_fields_map = invert_footprints(footprints, n_pixels)

    if adaptive_exptime:
        if mission.detector is None:
            raise NotImplementedError("This mission does not define a detector model")
        with status("evaluating exposure time map"):
            if appmag_dist:
                distmean, diststd, _ = distance.parameters_to_moments(
                    skymap_flat["DISTMU"],
                    skymap_flat["DISTSIGMA"],
                )
                logdist_sigma2 = np.log1p(np.square(diststd / distmean))
                logdist_sigma = np.sqrt(logdist_sigma2)
                logdist_mu = np.log(distmean) - 0.5 * logdist_sigma2
                a = 5 / np.log(10)
                appmag_mu = absmag_mean + a * logdist_mu + 25
                appmag_sigma = np.sqrt(
                    np.square(absmag_stdev) + np.square(a * logdist_sigma)
                )
                quantiles = np.linspace(0.05, 0.95, 5)
                appmag_quantiles = stats.norm(
                    loc=appmag_mu[:, np.newaxis], scale=appmag_sigma[:, np.newaxis]
                ).ppf(quantiles)
                # FIXME: prune pixels with infinite distance
                appmag_quantiles[np.isposinf(appmag_mu)] = np.inf

                with observing(
                    observer_location=observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good)[:, np.newaxis],
                    obstime=obstimes[0],
                ):
                    exptime_pixel_s = mission.detector.get_exptime(
                        snr,
                        synphot.SourceSpectrum(synphot.ConstFlux1D(0 * u.ABmag))
                        * synphot.SpectralElement(
                            TabularScaleFactor(
                                (
                                    appmag_quantiles * u.mag(u.dimensionless_unscaled)
                                ).to_value(u.dimensionless_unscaled)
                            )
                        )
                        * DustExtinction(),
                        visit_bandpasses[0],
                    ).to_value(u.s)
                exptime_max_s = max(
                    min(
                        exptime_max.to_value(u.s),
                        deadline.to_value(u.s),
                    ),
                    exptime_min_s,
                )
                piecewise_breakpoints = np.pad(
                    np.stack(
                        (
                            np.tile(quantiles[np.newaxis, :], (len(skymap_flat), 1)),
                            exptime_pixel_s,
                        ),
                        axis=-1,
                    ),
                    [(0, 0), (1, 0), (0, 0)],
                )
            else:
                distmod = Distance(skymap_moc.meta["distmean"] * u.Mpc).distmod
                with observing(
                    observer_location=observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good),
                    obstime=obstimes[0],
                ):
                    exptime_pixel_s = mission.detector.get_exptime(
                        snr,
                        synphot.SourceSpectrum(
                            synphot.ConstFlux1D(absmag_mean * u.ABmag + distmod)
                        )
                        * DustExtinction(),
                        visit_bandpasses[0],
                    ).to_value(u.s)
                exptime_min_s = min(
                    max(exptime_min_s, exptime_pixel_s.min(initial=exptime_min_s)),
                    exptime_max.to_value(u.s),
                )
                exptime_max_s = max(
                    min(
                        exptime_max.to_value(u.s),
                        deadline.to_value(u.s),
                        exptime_pixel_s.max(initial=exptime_max.to_value(u.s)),
                    ),
                    exptime_min_s,
                )

    with status("calculating slew times"):
        slew_i, slew_j = np.triu_indices(n_fields, 1)
        slew_time_s = mission.slew.time(
            target_coords[slew_i],
            target_coords[slew_j],
            rolls[slew_i],
            rolls[slew_j],
        ).to_value(u.s)

    with Model(
        timelimit=timelimit, jobs=jobs, memory=memory, lowercutoff=cutoff
    ) as model:
        with status("assembling MILP model"):
            if adaptive_exptime and appmag_dist:
                pixel_vars = model.continuous_vars(
                    n_pixels,
                    lb=0,
                    ub=[
                        breakpoints[(breakpoints[:, 1] < LARGE_EXPTIME), 0].max()
                        for breakpoints in piecewise_breakpoints
                    ],
                )
            else:
                pixel_vars = model.binary_vars(n_pixels)
            field_vars = model.binary_vars(n_fields)
            time_field_visit_vars = model.continuous_vars(
                (n_fields, visits),
            )
            if adaptive_exptime:
                exptime_field_vars = (
                    model.semicontinuous_vars
                    if exptime_min_s > 0
                    else model.continuous_vars
                )(n_fields, lb=exptime_min_s, ub=exptime_max_s)
                exptime_region_vars = model.continuous_vars(n_regions)

            # Add constraints on observability windows for each field
            with status("adding field of regard constraints"):
                for time_visit_vars, exptime, intervals in zip(
                    time_field_visit_vars,
                    np.tile(exptime_field_vars[:, np.newaxis], visits)
                    if adaptive_exptime
                    else np.tile(visit_exptime_min_s, (n_fields, 1)),
                    observable_intervals,
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

            if visits > 1:
                with status("adding cadence constraints"):
                    if adaptive_exptime:
                        rhs = (cadence_s * field_vars + exptime_field_vars)[
                            :, np.newaxis
                        ]
                    else:
                        rhs = np.multiply.outer(
                            field_vars, cadence_s + mean_consecutive_exptime_s
                        )
                    model.add_constraints_(
                        (time_field_visit_vars[:, 1:] - time_field_visit_vars[:, :-1])
                        >= rhs
                    )

            with status("adding slew constraints"):
                # Zero or less unless both fields are observed, which relaxes
                # the constraint away for any pair that is not.
                both_observed = field_vars[slew_i] + field_vars[slew_j] - 1
                if adaptive_exptime:
                    rhs = (
                        0.5 * (exptime_field_vars[slew_i] + exptime_field_vars[slew_j])
                        + slew_time_s * both_observed
                    )
                    rhs_within = rhs
                    rhs_after = rhs
                else:
                    # Two observations also clear each other by the slew itself.
                    def _spacing(exptimes):
                        return (
                            slew_time_s[np.newaxis, :] + exptimes[:, np.newaxis]
                        ) * both_observed[np.newaxis, :]

                    rhs = rhs_within = _spacing(visit_exptime_min_s)
                    rhs_after = _spacing(mean_consecutive_exptime_s)

                if any(filter_changes):
                    # Every field is visited for the kth time before any field
                    # is visited for the k+1th, so the filter is exchanged once
                    # per block boundary however many fields are observed. The
                    # ordering also makes the absolute value redundant across
                    # visits, leaving it only within one.
                    exchange_s = mission.filter_exchange_time.to_value(u.s)
                    gap = (
                        rhs_after
                        + exchange_s * np.asarray(filter_changes)[:, np.newaxis]
                    )
                    within_visit = (
                        time_field_visit_vars[slew_i, :]
                        - time_field_visit_vars[slew_j, :]
                    )
                    after_i = (
                        time_field_visit_vars[slew_i, 1:]
                        - time_field_visit_vars[slew_j, :-1]
                    )
                    after_j = (
                        time_field_visit_vars[slew_j, 1:]
                        - time_field_visit_vars[slew_i, :-1]
                    )
                    model.add_constraints_(
                        model.abs(np.transpose(within_visit)) >= rhs_within
                    )
                    model.add_constraints_(np.transpose(after_i) >= gap)
                    model.add_constraints_(np.transpose(after_j) >= gap)
                else:
                    p, q = full_indices(visits)
                    if not adaptive_exptime:
                        rhs = _spacing(
                            0.5 * (visit_exptime_min_s[p] + visit_exptime_min_s[q])
                        )
                    model.add_constraints_(
                        model.abs(
                            time_field_visit_vars[slew_i, p[:, np.newaxis]]
                            - time_field_visit_vars[slew_j, q[:, np.newaxis]]
                        )
                        >= rhs
                    )

            if adaptive_exptime:
                with status("adding exposure time constraints"):
                    model.add_constraints_(
                        exptime_max_s * field_vars >= exptime_field_vars
                    )

            with status("adding coverage constraints"):
                if adaptive_exptime:
                    if appmag_dist:
                        for pixel_var, region_index, breakpoints in zip(
                            pixel_vars, pixel_to_region_map, piecewise_breakpoints
                        ):
                            breakpoints = prepare_piecewise_breakpoints(breakpoints)
                            if len(breakpoints) <= 1:
                                assert pixel_var.ub == 0
                            else:
                                model.add_constraint_(
                                    exptime_region_vars[region_index]
                                    >= model.piecewise(0, breakpoints, 0)(pixel_var)
                                )
                    else:
                        model.add_indicators(
                            pixel_vars,
                            [
                                exptime_region_vars[region] >= exptime_s
                                for region, exptime_s in zip(
                                    pixel_to_region_map, exptime_pixel_s
                                )
                            ],
                        )
                    model.add_constraints_(
                        [
                            model.max(*exptime_field_vars[field_indices]).item()
                            >= exptime_var
                            for field_indices, exptime_var in zip(
                                region_to_fields_map, exptime_region_vars
                            )
                        ]
                    )
                else:
                    model.add_constraints_(
                        pixel_vars
                        <= [
                            model.sum_vars_all_different(field_vars[field_indices])
                            for field_indices in pixels_to_fields_map
                        ]
                    )

            with status("adding cuts"):
                model.add_user_cut_constraint(
                    model.sum_vars_all_different(field_vars)
                    <= (deadline - delay).to_value(u.s) / visit_exptime_min_s.sum()
                )
                if adaptive_exptime:
                    model.add_user_cut_constraint(
                        model.sum_vars_all_different(exptime_field_vars)
                        <= (deadline - delay).to_value(u.s) / visits
                    )

            with status("adding objective function"):
                model.maximize(
                    model.scal_prod_vars_all_different(pixel_vars, skymap_flat["PROB"])
                )

        if has_model := (
            model.number_of_constraints + model.objective_expr.number_of_terms() > 0
        ):
            with status("solving MILP model"):
                if write_progress is not None:
                    model.add_progress_listener(recorder := ProgressDataRecorder())

                if write_model is not None:
                    model.to_stream(write_model)

                solution = model.solve()
        else:
            solution = None

        with status("writing results"):
            if write_progress is not None:
                QTable(
                    rows=recorder.recorded,
                    names=ProgressData._fields,
                    dtype=[int, bool, float, float, float, int, int, int, float, float],
                ).write(write_progress, format="ascii.ecsv", overwrite=True)

            if solution is None:
                field_values = np.zeros(field_vars.shape, dtype=bool)
                time_field_visit_values = np.empty(time_field_visit_vars.shape)
                exptime_field_values = np.empty(time_field_visit_vars.shape)
                objective_value = 0.0
            else:
                field_values = solution.get_values(field_vars) >= 0.5
                time_field_visit_values = solution.get_values(time_field_visit_vars)
                if adaptive_exptime:
                    exptime_per_field = solution.get_values(exptime_field_vars)
                    field_values &= exptime_per_field > 0
                    exptime_field_values = np.tile(
                        exptime_per_field[:, np.newaxis], visits
                    )
                else:
                    exptime_field_values = np.tile(visit_exptime_min_s, (n_fields, 1))
                objective_value = solution.get_objective_value()

            table = QTable(
                {
                    "action": np.full(field_values.sum() * visits, "observe"),
                    "start_time": obstimes[0]
                    + (
                        time_field_visit_values[field_values]
                        - 0.5 * exptime_field_values[field_values]
                    ).ravel()
                    * u.s,
                    "duration": exptime_field_values[field_values].ravel() * u.s,
                    "target_coord": target_coords[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "roll": rolls[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "field_id": field_ids[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "bandpass": np.tile(
                        np.array([band or "" for band in visit_bandpasses], dtype=str),
                        field_values.sum(),
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
                },
                meta={
                    "command": shlex.join(sys.argv),
                    "version": __version__,
                    "args": {
                        "deadline": deadline,
                        "delay": delay,
                        "mission": mission.name,
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
                        "bandpass": visit_bandpasses,
                        "snr": snr,
                        "cutoff": cutoff,
                    },
                    "objective_value": objective_value,
                    "best_bound": model.best_bound if has_model else 0,
                    "solution_status": model.solve_details.status
                    if has_model
                    else "infeasible, no observable fields or pixels",
                    "solution_time": (model.solve_details.time if has_model else 0)
                    * u.s,
                },
            )
            table.sort("start_time")

            # Add orbit to table
            table.add_column(
                mission.observer_location(table["start_time"]),
                index=3,
                name="observer_location",
            )
            table["observer_location"].info.description = "Position of the spacecraft"

            # Add slew segments to table.
            if len(table) > 0:
                nrows = len(table) - 1
                slew_duration = mission.slew.time(
                    table["target_coord"][:-1],
                    table["target_coord"][1:],
                    table["roll"][:-1],
                    table["roll"][1:],
                )
                # The filter is exchanged while the telescope slews, so a
                # change costs only the excess over the slew itself.
                changed = table["bandpass"][:-1] != table["bandpass"][1:]
                slew_duration[changed] = np.maximum(
                    slew_duration[changed], mission.filter_exchange_time
                )
                slew_table = QTable(
                    {
                        "action": np.full(nrows, "slew"),
                        "start_time": (table["start_time"] + table["duration"])[:-1],
                        "duration": slew_duration,
                        "bandpass": np.full(nrows, ""),
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
                str(row["action"]): row["duration"].to(u.s)
                for row in total_time_by_action
            }
            table.meta["total_time"]["slack"] = (
                deadline - delay - total_time_by_action["duration"].sum()
            ).to(u.s)

            table.write(schedule, format="ascii.ecsv", overwrite=True)
