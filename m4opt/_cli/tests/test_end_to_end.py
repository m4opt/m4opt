from importlib import resources

import pytest
from astropy import units as u
from astropy.table import QTable, unique

from .. import app
from . import data


@pytest.fixture
def fits_path():
    with resources.path(data, "800.fits") as path:
        yield str(path)


@pytest.fixture
def ecsv_path(tmp_path):
    return tmp_path / "example.ecsv"


@pytest.fixture
def gif_path(tmp_path):
    return tmp_path / "example.gif"


@pytest.fixture(params=[None, -14])
def run_scheduler(fits_path, ecsv_path, gif_path, run_cli, request):
    absmag_mean = request.param

    def func(*args):
        args = [
            *args,
            "--bandpass=NUV",
            "--nside=128",
            "--deadline=6hour",
        ]
        if absmag_mean is not None:
            args = [*args, f"--absmag-mean={absmag_mean}"]
        result = run_cli(app, "schedule", fits_path, ecsv_path, *args)
        assert result.exit_code == 0
        table = QTable.read(ecsv_path)

        start_time_diff = table["start_time"][1:] - table["start_time"][:-1]

        assert (start_time_diff >= 0 * u.s).all(), "time intervals must be monotonic"
        assert (start_time_diff - table["duration"][:-1] >= -1e-3 * u.s).all(), (
            "time intervals must be non-overlapping"
        )

        assert (table["action"][::2] == "observe").all(), (
            "even actions must be 'observe'"
        )
        assert (table["action"][1::2] == "slew").all(), "odd actions must be 'slew'"

        observations = table[table["action"] == "observe"]
        num_fields = len(unique(observations["target_coord"].to_table()))
        num_visits = table.meta["args"]["visits"]
        assert len(observations) == num_visits * num_fields, (
            f"there are {num_fields} observations of each field"
        )

        assert (
            observations["duration"] + 1e-3 * u.s >= table.meta["args"]["exptime_min"]
        ).all()
        assert (observations["duration"] <= table.meta["args"]["exptime_max"]).all()

        result = run_cli(
            app,
            "animate",
            ecsv_path,
            gif_path,
            "--time-step=8hour",
            "--inset-center=35d -31d",
            "--inset-radius=11deg",
        )
        assert result.exit_code == 0
        assert gif_path.read_bytes().startswith(b"GIF89a")
        return table

    return func


def test_end_to_end_no_solution(run_scheduler):
    table = run_scheduler("--timelimit=1s", "--exptime-min=5hour", "--cutoff=0.1")
    assert len(table) == 0
    assert table.meta["solution_status"].startswith("aborted")
    assert table.meta["objective_value"] == pytest.approx(0, abs=1e-7)
    assert table.meta["total_time"]["slack"] == 6 * u.hour


def test_end_to_end_solution(run_scheduler):
    table = run_scheduler("--timelimit=1min", "--exptime-min=300s")
    assert len(table) >= 3
