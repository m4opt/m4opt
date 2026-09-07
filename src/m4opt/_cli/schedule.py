import shlex
import sys
from itertools import pairwise
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
        np.asarray(fields, dtype=np.intp) for fields in region_to_fields_map
    ]
    return pixel_to_region_map, region_to_fields_map


LARGE_EXPTIME = 1e10


def prepare_piecewise_breakpoints(breakpoints):
    isinf_indices = np.flatnonzero(breakpoints[:, 1] >= LARGE_EXPTIME)
    if len(isinf_indices) > 0:
        breakpoints = breakpoints[: isinf_indices[0]]
    return [tuple(col.item() for col in row) for row in breakpoints]


def add_filter_block_constraints(
    model, field_vars, time_field_visit_vars, half_exptime, filter_changes, exchange_s
):
    """Confine each visit to a contiguous block of a single bandpass.

    Every field observes visit ``i`` in the same bandpass, so grouping the
    visits into non-overlapping blocks means that the filter is exchanged once
    per element of ``filter_changes`` that is true, however many fields are
    observed.

    The exchange is assumed to happen while the telescope slews, so the gap
    between two blocks is the exchange time rather than the exchange time plus
    a slew.

    Parameters
    ----------
    model
        The MILP model.
    field_vars
        Binary variable for each field, true if the field is observed.
    time_field_visit_vars
        Midpoint time of each visit to each field, in seconds.
    half_exptime
        Half of the exposure time of each field, in seconds. May be an array of
        variables if the exposure time is adaptive.
    filter_changes
        For each pair of consecutive visits, whether the bandpass changes.
    exchange_s
        Time to exchange one filter for another, in seconds.
    """
    visits = time_field_visit_vars.shape[1]
    block_start_vars = model.continuous_vars(visits)
    block_end_vars = model.continuous_vars(visits)
    for visit in range(visits):
        times = time_field_visit_vars[:, visit]
        model.add_indicators(
            field_vars, times - half_exptime - block_start_vars[visit] >= 0
        )
        model.add_indicators(
            field_vars, times + half_exptime - block_end_vars[visit] <= 0
        )
    model.add_constraints_(
        block_start_vars[1:]
        - block_end_vars[:-1]
        - exchange_s * np.asarray(filter_changes)
        >= 0
    )
    return block_start_vars, block_end_vars


def _exptime_over_bandpasses(mission, snr, spectrum, bandpasses):
    """Exposure time reaching the target SNR in every one of ``bandpasses``.

    A field has a single exposure time shared by all of its visits, so the
    least sensitive bandpass governs.
    """
    return np.maximum.reduce(
        [
            mission.detector.get_exptime(snr, spectrum, bandpass).to_value(u.s)
            for bandpass in bandpasses
        ]
    )


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
    filter_change: Annotated[
        bool, typer.Option(help="If the telescope has lengthy filter changes.")
    ] = False,
    visits: Annotated[int, typer.Option(min=1, help="Number of visits")] = 2,
    cadence: Annotated[
        u.Quantity,
        typer.Option(help="Minimum time separation between visits"),
    ] = 30 * u.min,
    time_seqs: Annotated[
        typer.FileText | None,
        typer.Option(
            help="Optional CSV file with a preset sequence of exposure times (line 1, in s) and inter-round delays (line 2, in min) for a telescope with filter changes."
        ),
    ] = None,
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
    The scheduler has five modes:

    \b
    1. Fixed exposure time. Every field has the same exposure time given by the
       --exptime-min option. This mode is selected if you omit the
       --absmag-mean option.

    \b
    2. Fixed exposure time with filter changes. Every field has the same
        exposure time given by the --exptime-min option. Each field is visited
        k times before any field is visited k+1 times. This mode is selected
        if you omit the --absmag-mean option and choose --filter-change.

    \b
    3. Pre-set exposure times and inter-round delays. Exposure times and
        delays between consecutive visits of the same field are pre-set in a
        provided CSV file. All fields on their k visit will have the k exposure
        time specified in the CSV file. Consecutive observations of the same
        field (e.g. k and k+1 visits) will be separated by the k delay
        specified in the CSV file. Each field is visited k times before any
        field is visited k+1 times. This mode is selected if you choose
        --filter-change and provide a file for the --test-seqs option.

    \b
    4. Variable exposure time. Each field may have a different exposure time,
       adjusted for the posterior median distance along each line of sight.
       This mode is selected if you specify a value for the --absmag-mean
       option but also pass the --no-appmag-dist option.

    \b
    5. Variable exposure time with an absolute magnitude distribution. Each
       field may have a different exposure time, adjusted to optimize the
       detection probability given the posterior distance distribution and a
       Gaussian distribution of absolute magnitudes. This mode is selected if
       you specify the --absmag-mean option (and, optionally, the
       --absmag-stdev option).
    """
    adaptive_exptime = absmag_mean is not None
    preset_exp_del = time_seqs is not None
    if (filter_change is False) and preset_exp_del:
        raise NotImplementedError(
            "Sequences of exposure times and inter-round delays can only be used if the telescope has filter changes. Please pass --filter-change."
        )
    if adaptive_exptime and filter_change:
        raise NotImplementedError(
            "Adaptive exposure time cannot currently be used if the telescope has filter changes. Please choose one to implement."
        )

    # Successive visits cycle through the requested bandpasses, so that
    # --bandpass g --bandpass r over three visits gives g, r, g.
    visit_bandpasses = [
        bandpass[i % len(bandpass)] if bandpass else None for i in range(visits)
    ]
    unique_bandpasses = list(dict.fromkeys(visit_bandpasses))
    filter_changes = [lhs != rhs for lhs, rhs in pairwise(visit_bandpasses)]

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

        # Extract exposure time and inter-round delays from CSV file
        if preset_exp_del:
            time_seqs_lines = time_seqs.readlines()
            exptime_seq = time_seqs_lines[0].split(",")
            exptime_seq = np.asarray([float(x) for x in exptime_seq])
            cadence_seq = time_seqs_lines[1].split(",")
            cadence_seq = np.asarray([float(y) for y in cadence_seq]) * u.min
            cadence_seq = cadence_seq.to_value(u.s)
            assert len(exptime_seq) == (len(cadence_seq) + 1), (
                "Number of exposure times and number of inter-round delays must be consistent."
            )
            assert len(exptime_seq) == visits, (
                "Exactly one exposure time must be provided for each desired visit."
            )
            assert (len(cadence_seq) + 1) == visits, (
                "Number of inter-round delays must be exactly one less than the number of visits."
            )
            assert np.all(exptime_seq >= exptime_min_s), (
                "All values of exposure time must exceed the specified minimum exposure time."
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
                    observer_location=observer_locations
                    if isinstance(mission.observer_location, EarthFixedObserverLocation)
                    else observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good)[:, np.newaxis],
                    obstime=obstimes[0],
                ):
                    spectrum = (
                        synphot.SourceSpectrum(synphot.ConstFlux1D(0 * u.ABmag))
                        * synphot.SpectralElement(
                            TabularScaleFactor(
                                (
                                    appmag_quantiles * u.mag(u.dimensionless_unscaled)
                                ).to_value(u.dimensionless_unscaled)
                            )
                        )
                        * DustExtinction()
                    )
                    exptime_pixel_s = _exptime_over_bandpasses(
                        mission, snr, spectrum, unique_bandpasses
                    )
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
                    observer_location=observer_locations
                    if isinstance(mission.observer_location, EarthFixedObserverLocation)
                    else observer_locations[0],
                    target_coord=hpx.healpix_to_skycoord(good),
                    obstime=obstimes[0],
                ):
                    exptime_pixel_s = _exptime_over_bandpasses(
                        mission,
                        snr,
                        synphot.SourceSpectrum(
                            synphot.ConstFlux1D(absmag_mean * u.ABmag + distmod)
                        )
                        * DustExtinction(),
                        unique_bandpasses,
                    )
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
                    exptime_field_vars
                    if adaptive_exptime
                    else np.full(n_fields, exptime_min_s),
                    observable_intervals,
                ):
                    assert len(intervals) > 0
                    begin, end = intervals.T
                    if preset_exp_del:
                        exptime = exptime_seq
                    if len(intervals) == 1:
                        model.add_constraints_(
                            time_visit_vars - begin - 0.5 * exptime >= 0
                        )
                        model.add_constraints_(
                            time_visit_vars - end + 0.5 * exptime <= 0
                        )
                    else:
                        if preset_exp_del:
                            exptime = exptime[:, np.newaxis]
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
                    if preset_exp_del:
                        exptime_del_buffer = (
                            0.5 * (exptime_seq[1:] + exptime_seq[:-1]) + cadence_seq
                        )
                        rhs = (
                            field_vars[:, np.newaxis]
                            * exptime_del_buffer[np.newaxis, :]
                        )
                    else:
                        if adaptive_exptime:
                            rhs = cadence_s * field_vars + exptime_field_vars
                        else:
                            rhs = (exptime_min_s + cadence_s) * field_vars
                        rhs = rhs[:, np.newaxis]
                    model.add_constraints_(
                        (time_field_visit_vars[:, 1:] - time_field_visit_vars[:, :-1])
                        >= rhs
                    )

            if any(filter_changes):
                with status("adding filter block constraints"):
                    add_filter_block_constraints(
                        model,
                        field_vars,
                        time_field_visit_vars,
                        0.5 * exptime_field_vars
                        if adaptive_exptime
                        else np.full(n_fields, 0.5 * exptime_min_s),
                        filter_changes,
                        mission.filter_exchange_time.to_value(u.s),
                    )

            with status("adding slew constraints"):
                p, q = full_indices(visits)
                if preset_exp_del:
                    rhs1 = (slew_time_s[:, np.newaxis] + exptime_seq[np.newaxis, :]) * (
                        field_vars[slew_i, np.newaxis]
                        + field_vars[slew_j, np.newaxis]
                        - 1
                    )
                    rhs2 = (
                        slew_time_s[:, np.newaxis]
                        + 0.5
                        * (exptime_seq[np.newaxis, 1:] + exptime_seq[np.newaxis, :-1])
                    ) * (
                        field_vars[slew_i, np.newaxis]
                        + field_vars[slew_j, np.newaxis]
                        - 1
                    )
                    rhs3 = (
                        slew_time_s[:, np.newaxis]
                        + 0.5
                        * (exptime_seq[np.newaxis, 1:] + exptime_seq[np.newaxis, :-1])
                    ) * (
                        field_vars[slew_j, np.newaxis]
                        + field_vars[slew_i, np.newaxis]
                        - 1
                    )
                elif adaptive_exptime:
                    rhs = 0.5 * (
                        exptime_field_vars[slew_i] + exptime_field_vars[slew_j]
                    ) + slew_time_s * (field_vars[slew_i] + field_vars[slew_j] - 1)
                else:
                    rhs = (slew_time_s + exptime_min_s) * (
                        field_vars[slew_i] + field_vars[slew_j] - 1
                    )
                    if filter_change:
                        rhs1 = rhs
                        rhs2 = rhs
                        rhs3 = (slew_time_s + exptime_min_s) * (
                            field_vars[slew_j] + field_vars[slew_i] - 1
                        )
                if filter_change:
                    same_visit_times = (
                        time_field_visit_vars[slew_i, :]
                        - time_field_visit_vars[slew_j, :]
                    )
                    consec_visit_times1 = (
                        time_field_visit_vars[slew_i, 1:]
                        - time_field_visit_vars[slew_j, :-1]
                    )
                    consec_visit_times2 = (
                        time_field_visit_vars[slew_j, 1:]
                        - time_field_visit_vars[slew_i, :-1]
                    )
                    if preset_exp_del:
                        model.add_constraints_(model.abs(same_visit_times) >= rhs1)
                        model.add_constraints_(consec_visit_times1 >= rhs2)
                        model.add_constraints_(consec_visit_times2 >= rhs3)
                    else:
                        model.add_constraints_(
                            model.abs(np.transpose(same_visit_times)) >= rhs1
                        )
                        model.add_constraints_(
                            np.transpose(consec_visit_times1) >= rhs2
                        )
                        model.add_constraints_(
                            np.transpose(consec_visit_times2) >= rhs3
                        )
                else:
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
                if preset_exp_del:
                    exptime_sum = np.sum(exptime_seq)
                else:
                    exptime_sum = visits * exptime_min_s
                model.add_user_cut_constraint(
                    model.sum_vars_all_different(field_vars)
                    <= (deadline - delay).to_value(u.s) / (exptime_sum)
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
                exptime_field_values = np.empty(field_vars.shape)
                objective_value = 0.0
                start_time_add = (
                    time_field_visit_values[field_values]
                    - 0.5 * exptime_field_values[field_values][:, np.newaxis]
                )
                tiled_duration = np.tile(
                    exptime_field_values[field_values][:, np.newaxis], visits
                )
            else:
                field_values = solution.get_values(field_vars) >= 0.5
                time_field_visit_values = solution.get_values(time_field_visit_vars)
                if preset_exp_del:
                    start_time_add = (
                        time_field_visit_values[field_values] - 0.5 * exptime_seq
                    )
                    tiled_duration = np.tile(exptime_seq, field_values.sum())
                else:
                    if adaptive_exptime:
                        exptime_field_values = solution.get_values(exptime_field_vars)
                        field_values &= exptime_field_values > 0
                    else:
                        exptime_field_values = np.full(n_fields, exptime_min_s)
                    start_time_add = (
                        time_field_visit_values[field_values]
                        - 0.5 * exptime_field_values[field_values][:, np.newaxis]
                    )
                    tiled_duration = np.tile(
                        exptime_field_values[field_values][:, np.newaxis], visits
                    )
                objective_value = solution.get_objective_value()

            table = QTable(
                {
                    "action": np.full(field_values.sum() * visits, "observe"),
                    "start_time": obstimes[0] + start_time_add.ravel() * u.s,
                    "duration": tiled_duration.ravel() * u.s,
                    "target_coord": target_coords[
                        np.tile(np.flatnonzero(field_values)[:, np.newaxis], visits)
                    ].ravel(),
                    "roll": rolls[
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
                        "time_step": time_step,
                        "skymap": skymap.name,
                        "visits": visits,
                        "exptime_min": exptime_min,
                        "exptime_max": exptime_max,
                        "filter-change": filter_change,
                        "time-seqs": time_seqs.name if time_seqs is not None else None,
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
