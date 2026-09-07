"""Tests for scheduling visits in per-bandpass blocks."""

from importlib import resources
from itertools import pairwise

import numpy as np
import pytest
from astropy.table import QTable

from .. import app
from . import data


@pytest.fixture
def fits_path():
    with resources.path(data, "800.fits") as path:
        yield str(path)


@pytest.fixture
def schedule(fits_path, tmp_path, run_cli):
    """A UVEX schedule of three visits alternating between two bandpasses."""
    out = tmp_path / "blocks.ecsv"
    result = run_cli(
        app,
        "schedule",
        fits_path,
        out,
        "--mission=uvex",
        "--bandpass=NUV",
        "--bandpass=FUV",
        "--visits=3",
        "--nside=64",
        "--deadline=8hour",
        "--timelimit=60s",
        "--exptime-min=300s",
        "--no-appmag-dist",
    )
    assert result.exit_code == 0
    table = QTable.read(out)
    return table[table["action"] == "observe"]


def test_visits_are_grouped_into_blocks(schedule):
    """Every field is visited for the kth time before any is visited again."""
    if len(schedule) == 0:
        pytest.skip("no fields were observable")
    schedule.sort("start_time")
    bands = list(schedule["bandpass"])
    # Contiguous runs of one bandpass, one run per visit.
    runs = [band for i, band in enumerate(bands) if i == 0 or band != bands[i - 1]]
    assert len(runs) == 3
    assert runs == ["NUV", "FUV", "NUV"]


def test_one_filter_change_per_block_boundary(schedule):
    """The schedule exchanges the filter once per boundary, not once per field."""
    if len(schedule) == 0:
        pytest.skip("no fields were observable")
    schedule.sort("start_time")
    bands = list(schedule["bandpass"])
    changes = sum(a != b for a, b in pairwise(bands))
    assert changes == 2
    assert len(schedule) > changes


def test_a_single_bandpass_needs_no_ordering(fits_path, tmp_path, run_cli):
    """One bandpass throughout leaves the visit ordering unconstrained."""
    out = tmp_path / "single.ecsv"
    result = run_cli(
        app,
        "schedule",
        fits_path,
        out,
        "--mission=uvex",
        "--bandpass=NUV",
        "--visits=2",
        "--nside=64",
        "--deadline=8hour",
        "--timelimit=60s",
        "--exptime-min=300s",
        "--no-appmag-dist",
    )
    assert result.exit_code == 0
    table = QTable.read(out)
    observations = table[table["action"] == "observe"]
    assert set(np.asarray(observations["bandpass"])) <= {"NUV"}
