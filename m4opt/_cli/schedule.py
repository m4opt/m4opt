import gzip
import shlex
import sys
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Annotated

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

from .. import __version__, missions
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
        np.asarray(fields, dtype=np.intp) for fields in region_to_fields_map.keys()
    ]
    return pixel_to_region_map, region_to_fields_map


LARGE_EXPTIME = 1e10


def prepare_piecewise_breakpoints(breakpoints):
    isinf_indices = np.flatnonzero(breakpoints[:, 1] >= LARGE_EXPTIME)
    if len(isinf_indices) > 0:
        breakpoints = breakpoints[: isinf_indices[0]]
    return [tuple(col.item() for col in row) for row in breakpoints]


def write_model_to_stream(model: Model, out_file: typer.FileBinaryWrite):
    valid_formats = {"lp", "mps", "sav"}
    out_filename = out_file.name
    out_path = Path(out_filename)
    suffixes = Path(out_path).suffixes

    if (
        len(suffixes) == 0
        or (format := suffixes[0].lstrip(".").lower()) not in valid_formats
    ):
        valid_extensions = [f".{fmt}" for fmt in valid_formats]
        valid_extensions = [
            *valid_extensions,
            *(f"{ext}.gz" for ext in valid_extensions),
        ]
        raise typer.BadParameter(
            f'Invalid model filename "{out_filename}". The extension must be one of the following: {" ".join(valid_extensions)}'
        )
    export_method = getattr(model, f"export_as_{format}")

    should_gzip = suffixes[-1].lower() == ".gz"

    if should_gzip:
        with NamedTemporaryFile(suffix=f".{format}") as temp_file:
            export_method(temp_file.name)
            with gzip.GzipFile(
                f"{out_path.name}{suffixes[0]}", "wb", fileobj=out_file
            ) as zip_file:
                copyfileobj(temp_file, zip_file)
    else:
        export_method(out_filename)


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
    mission: Annotated[
        missions.Mission, typer.Option(show_default="uvex")
    ] = missions.uvex,
    skygrid: Annotated[
        str | None,
        typer.Option(
            help="Name of sky grid to use, if the mission supports multiple sky grids.",
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
    exptime_min: Annotated[
        u.Quantity,
        typer.Option(
            help="Minimum exposure time for each observation",
        ),
    ] = 900 * u.s,
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
        float | None,
        typer.Option(
            help="Standard deviation of AB absolute magnitude of source",
            show_default="AB absolute magnitude is fixed at the value provided by --absmag-mean",
        ),
    ] = None,
    snr: Annotated[float, typer.Option(help="Signal to noise ratio for detection")] = 5,
    bandpass: Annotated[
        str | None, typer.Option(help="Name of detector bandpass")
    ] = None,
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 2,
    cadence: Annotated[
        u.Quantity,
        typer.Option(help="Minimum time separation between visits"),
    ] = 30 * u.min,
    nside: Annotated[int, typer.Option(help="HEALPix resolution")] = 512,
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
    """Generate an observing plan for a GW sky map.

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
       option.

    \b
    3. Variable exposure time with an absolute magnitude distribution. Each
       field may have a different exposure time, adjusted to optimize the
       detection probability given the posterior distance distribution and a
       Gaussian distribution of absolute magnitudes. This mode is selected if
       you specify both the --absmag-mean and --absmag-stdev options.
    """
    adaptive_exptime = absmag_mean is not None
    absmag_distribution = absmag_stdev is not None

    """Schedule a target of opportunity observation."""
    with status("loading sky map"):
        hpx = HEALPix(nside, frame=ICRS(), order="nested")
        skymap_moc = read_sky_map(skymap, moc=True)
        skymap_flat = rasterize(skymap_moc, hpx.level)
        event_time = Time(
            Time(skymap_moc.meta["gps_time"], format="gps").utc, format="iso"
        )

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

        # FIXME: https://github.com/astropy/astropy/issues/17030
        target_coords = SkyCoord(target_coords.ra, target_coords.dec)
        exptime_min_s = exptime_min.to_value(u.s)
        cadence_s = cadence.to_value(u.s)
        obstimes_s = (obstimes - obstimes[0]).to_value(u.s)
        observable_intervals = np.asarray(
            [
                obstimes_s[intervals]
                for intervals in clump_nonzero_inclusive(
                    mission.constraints(
                        observer_locations,
                        target_coords[:, np.newaxis],
                        obstimes,
                    )
                )
            ],
            dtype=object,
        )

        # Keep only intervals that are at least as long as the exposure time.
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

        # Select only the most probable 50 fields.
        n_fields = 50
        if len(target_coords) > n_fields:
            good = np.argpartition(
                [-skymap_flat[footprint]["PROB"].sum() for footprint in footprints],
                n_fields,
            )[:n_fields]
            target_coords = target_coords[good]
            rolls = rolls[good]
            footprints = footprints[good]
            observable_intervals = observable_intervals[good]
        else:
            n_fields = len(target_coords)

        # Throw away pixels that are not contained in any fields.
        good = np.unique(np.concatenate(footprints))
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
            if (
                absmag_stdev is not None
            ):  # same as `if absmag_distribution:` but allows mypy to infer that absmag_stdev is not None
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
                        bandpass,
                    ).to_value(u.s)
                exptime_max_s = max(
                    min(
                        exptime_max.to_value(u.s),
                        deadline.to_value(u.s),
                    ),
                    exptime_min.to_value(u.s),
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
                        bandpass,
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
                    exptime_min.to_value(u.s),
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
            if absmag_distribution:
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
                    exptime_field_vars
                    if adaptive_exptime
                    else np.full(n_fields, exptime_min_s),
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
                        model.add_indicators(
                            visit_interval_vars,
                            time_visit_vars[:, np.newaxis] - begin - 0.5 * exptime >= 0,
                        )
                        model.add_indicators(
                            visit_interval_vars,
                            time_visit_vars[:, np.newaxis] - end + 0.5 * exptime <= 0,
                        )

            if visits > 1:
                with status("adding cadence constraints"):
                    if adaptive_exptime:
                        rhs = cadence_s * field_vars + exptime_field_vars
                    else:
                        rhs = (exptime_min_s + cadence_s) * field_vars
                    model.add_constraints_(
                        (time_field_visit_vars[:, 1:] - time_field_visit_vars[:, :-1])
                        >= rhs[:, np.newaxis]
                    )

            with status("adding slew constraints"):
                p, q = full_indices(visits)
                if adaptive_exptime:
                    rhs = 0.5 * (
                        exptime_field_vars[slew_i] + exptime_field_vars[slew_j]
                    ) + slew_time_s * (field_vars[slew_i] + field_vars[slew_j] - 1)
                else:
                    rhs = (slew_time_s + exptime_min_s) * (
                        field_vars[slew_i] + field_vars[slew_j] - 1
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
                    if absmag_distribution:
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
                    <= (deadline - delay).to_value(u.s) / (visits * exptime_min_s)
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

        with status("solving MILP model"):
            if write_progress is not None:
                model.add_progress_listener(recorder := ProgressDataRecorder())

            if write_model is not None:
                write_model_to_stream(model, write_model)

            solution = model.solve()

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
                exptime_field_values = np.empty(field_vars.shape)
                objective_value = 0.0
            else:
                field_values = solution.get_values(field_vars) >= 0.5
                time_field_visit_values = solution.get_values(time_field_visit_vars)
                if adaptive_exptime:
                    exptime_field_values = solution.get_values(exptime_field_vars)
                    field_values &= exptime_field_values > 0
                else:
                    exptime_field_values = np.full(n_fields, exptime_min_s)
                objective_value = solution.get_objective_value()

            table = QTable(
                {
                    "action": np.full(field_values.sum() * visits, "observe"),
                    "start_time": obstimes[0]
                    + (
                        time_field_visit_values[field_values]
                        - 0.5 * exptime_field_values[field_values][:, np.newaxis]
                    ).ravel()
                    * u.s,
                    "duration": np.tile(
                        exptime_field_values[field_values][:, np.newaxis], visits
                    ).ravel()
                    * u.s,
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
                        "skygrid": skygrid,
                        "nside": nside,
                        "time_step": time_step,
                        "skymap": skymap.name,
                        "visits": visits,
                        "exptime_min": exptime_min,
                        "exptime_max": exptime_max,
                        "absmag_mean": absmag_mean,
                        "absmag_stdev": absmag_stdev,
                        "bandpass": bandpass,
                        "snr": snr,
                        "cutoff": cutoff,
                    },
                    "objective_value": objective_value,
                    "best_bound": model.best_bound,
                    "solution_status": model.solve_details.status,
                    "solution_time": model.solve_details.time * u.s,
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
