"""Tests for scheduling visits in per-bandpass blocks."""

from importlib import resources
from itertools import pairwise

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from click import UsageError

from .. import app
from . import data


@pytest.fixture
def fits_path():
    with resources.path(data, "800.fits") as path:
        yield str(path)


def _observations(out):
    table = QTable.read(out)
    observations = table[table["action"] == "observe"]
    observations.sort("start_time")
    return observations


def test_visits_are_grouped_into_blocks(fits_path, tmp_path, run_cli):
    """
    Each visit is a contiguous block, costing one filter change per boundary.

    ZTF exchanges filters through a filter changer, so the order of the visits
    decides how many exchanges a schedule pays for.
    """
    out = tmp_path / "blocks.ecsv"
    result = run_cli(
        app,
        "schedule",
        fits_path,
        out,
        "--mission=ztf",
        "--bandpass=g",
        "--bandpass=r",
        "--visits=3",
        "--nside=16",
        "--max-fields=6",
        "--deadline=8hour",
        "--timelimit=60s",
        "--exptime-min=300s",
        "--no-appmag-dist",
    )
    assert result.exit_code == 0
    observations = _observations(out)
    if len(observations) == 0:
        pytest.skip("no fields were observable")

    bands = list(observations["bandpass"])
    runs = [band for i, band in enumerate(bands) if i == 0 or band != bands[i - 1]]
    assert runs == ["g", "r", "g"]

    # One exchange per boundary between blocks, not one per field.
    changes = sum(a != b for a, b in pairwise(bands))
    assert changes == len(runs) - 1
    assert len(observations) > changes


def test_a_single_bandpass_needs_no_ordering(fits_path, tmp_path, run_cli):
    """One bandpass throughout leaves the visit ordering unconstrained."""
    out = tmp_path / "single.ecsv"
    result = run_cli(
        app,
        "schedule",
        fits_path,
        out,
        "--mission=ztf",
        "--bandpass=g",
        "--visits=2",
        "--nside=16",
        "--max-fields=6",
        "--deadline=8hour",
        "--timelimit=60s",
        "--exptime-min=300s",
        "--no-appmag-dist",
    )
    assert result.exit_code == 0
    assert set(np.asarray(_observations(out)["bandpass"])) == {"g"}


def test_each_bandpass_may_have_its_own_exposure_time(fits_path, tmp_path, run_cli):
    """Repeating --exptime-min gives each bandpass the exposure time it needs."""
    out = tmp_path / "per_band.ecsv"
    result = run_cli(
        app,
        "schedule",
        fits_path,
        out,
        "--mission=ztf",
        "--bandpass=g",
        "--bandpass=r",
        "--visits=2",
        "--nside=16",
        "--max-fields=6",
        "--deadline=8hour",
        "--timelimit=60s",
        "--exptime-min=120s",
        "--exptime-min=300s",
        "--no-appmag-dist",
    )
    assert result.exit_code == 0
    observations = _observations(out)
    if len(observations) == 0:
        pytest.skip("no fields were observable")

    durations = {
        str(band): float(duration.to_value(u.s))
        for band, duration in zip(observations["bandpass"], observations["duration"])
    }
    assert durations == {"g": 120.0, "r": 300.0}

    # The longer exposure still has to fit between its neighbours.
    starts = observations["start_time"].gps
    ends = starts + observations["duration"].to_value(u.s)
    assert (starts[1:] >= ends[:-1]).all()


def test_one_exposure_time_per_bandpass_or_one_in_total(fits_path, tmp_path, run_cli):
    """Giving neither one exposure time nor one per bandpass is a usage error."""
    with pytest.raises(UsageError, match="one for every bandpass"):
        run_cli(
            app,
            "schedule",
            fits_path,
            tmp_path / "bad.ecsv",
            "--mission=ztf",
            "--bandpass=g",
            "--bandpass=r",
            "--visits=2",
            "--exptime-min=120s",
            "--exptime-min=300s",
            "--exptime-min=60s",
            "--no-appmag-dist",
        )


def test_a_variable_exposure_time_refuses_more_than_one_bandpass(
    fits_path, tmp_path, run_cli
):
    """Each field has one exposure time, so it cannot serve several bandpasses."""
    with pytest.raises(NotImplementedError, match="more than one"):
        run_cli(
            app,
            "schedule",
            fits_path,
            tmp_path / "adaptive.ecsv",
            "--mission=ztf",
            "--bandpass=g",
            "--bandpass=r",
            "--visits=2",
            "--exptime-min=300s",
            "--absmag-mean=-16",
        )
