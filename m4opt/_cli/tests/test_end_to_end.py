import pytest
from astropy import units as u
from astropy.table import QTable, unique
from astropy.utils.data import download_file

from .. import app


@pytest.fixture
def fits_path():
    url = "https://gracedb.ligo.org/api/superevents/S190425z/files/bayestar.fits"
    return download_file(url, cache=True)


@pytest.fixture
def ecsv_path(tmp_path):
    return tmp_path / "example.ecsv"


@pytest.fixture
def gif_path(tmp_path):
    return tmp_path / "example.gif"


@pytest.fixture(params=[None, -10])
def run_scheduler(fits_path, ecsv_path, gif_path, run_cli, request):
    absmag_mean = request.param

    def func(*args):
        if absmag_mean is not None:
            args = [*args, f"--absmag-mean={absmag_mean}", "--bandpass=NUV"]
        result = run_cli(app, "schedule", fits_path, ecsv_path, *args)
        assert result.exit_code == 0
        table = QTable.read(ecsv_path)

        start_time_diff = table["start_time"][1:] - table["start_time"][:-1]

        assert (start_time_diff > 0 * u.s).all(), "time intervals must be monotonic"
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

        assert (observations["duration"] >= table.meta["args"]["exptime_min"]).all()
        assert (observations["duration"] <= table.meta["args"]["exptime_max"]).all()

        result = run_cli(app, "animate", ecsv_path, gif_path, "--time-step=8hour")
        assert result.exit_code == 0
        assert gif_path.read_bytes().startswith(b"GIF89a")
        return table

    return func


def test_end_to_end_no_solution(run_scheduler):
    table = run_scheduler("--timelimit=1s")
    assert len(table) == 0
    assert table.meta["solution_status"].startswith("time limit exceeded")
    assert table.meta["objective_value"] == 0
    assert table.meta["total_time"]["slack"] == 1 * u.day


def test_end_to_end_solution(run_scheduler):
    table = run_scheduler("--deadline=2day", "--timelimit=60s")
    assert len(table) >= 3
