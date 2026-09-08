from importlib import resources

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable, unique
from click import UsageError

from ... import missions
from .. import app
from . import data

#: Each mission with a bandpass that its detector has, so that the same test
#: bodies run for a space telescope and a ground based one.
MISSIONS = {
    "uvex": ("--mission=uvex", "--bandpass=NUV"),
    "ztf": ("--mission=ztf", "--bandpass=g"),
}


@pytest.fixture(params=MISSIONS)
def mission_args(request):
    return MISSIONS[request.param]


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
def run_scheduler(fits_path, ecsv_path, gif_path, run_cli, mission_args, request):
    absmag_mean = request.param

    def func(*args):
        args = [
            *args,
            *mission_args,
            "--nside=128",
            "--deadline=6hour",
            "--no-appmag-dist",
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


def test_field_id_names_the_field_observed(run_scheduler):
    """Each observation names the field it points at, as the mission names it."""
    table = run_scheduler("--timelimit=1min", "--exptime-min=300s")
    observations = table[table["action"] == "observe"]
    assert len(observations) > 0

    mission = getattr(missions, table.meta["args"]["mission"])
    grid = mission.skygrid
    if isinstance(grid, dict):
        grid = grid[table.meta["args"]["skygrid"]]
    field_ids = np.asarray(observations["field_id"])
    assert np.all(field_ids >= 0)
    # The identifier indexes the grid, so no translation is needed.
    pointing = grid[field_ids]
    separation = pointing.separation(observations["target_coord"])
    np.testing.assert_allclose(np.asarray(separation.deg), 0, atol=1e-9)

    # A slew belongs to no field.
    assert np.all(np.asarray(table[table["action"] == "slew"]["field_id"]) == -1)


def test_fixed_exptime_with_appmag_dist(fits_path, ecsv_path, run_cli, mission_args):
    """Fixed exposure time mode should work when appmag_dist is True (default).

    Regression test for https://github.com/m4opt/m4opt/issues/XXX:
    When --absmag-mean is not provided (fixed exposure time) but appmag_dist
    defaults to True, the scheduler would crash with an UnboundLocalError
    accessing piecewise_breakpoints.
    """
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        *mission_args,
        "--nside=128",
        "--deadline=6hour",
        "--exptime-min=300s",
        "--timelimit=1s",
        # Notably: no --no-appmag-dist and no --absmag-mean
    )
    assert result.exit_code == 0


def test_max_fields_limits_the_problem(fits_path, ecsv_path, run_cli):
    """No more fields are scheduled than the cap allows."""
    max_fields = 3
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        "--mission=uvex",
        "--bandpass=NUV",
        "--nside=32",
        "--deadline=8hour",
        "--timelimit=30s",
        "--no-appmag-dist",
        f"--max-fields={max_fields}",
    )
    assert result.exit_code == 0
    table = QTable.read(ecsv_path)
    observations = table[table["action"] == "observe"]
    assert len(unique(observations["target_coord"].to_table())) <= max_fields
    assert table.meta["args"]["max_fields"] == max_fields


@pytest.fixture
def skymap_without_gps_time(tmp_path):
    """A sky map generated locally, which carries no trigger time."""
    import astropy_healpix as ah
    from ligo.skymap.io import write_sky_map

    path = str(tmp_path / "nogps.fits")
    npix = ah.nside_to_npix(8)
    write_sky_map(path, np.full(npix, 1 / npix), moc=False, nest=True)
    return path


def test_event_time_required_when_absent_from_sky_map(
    skymap_without_gps_time, ecsv_path, run_cli
):
    """A sky map with no trigger time says how to supply one."""
    with pytest.raises(UsageError, match="--event-time"):
        run_cli(app, "schedule", skymap_without_gps_time, ecsv_path, "--mission=uvex")


def test_event_time_option_supplies_the_trigger_time(
    skymap_without_gps_time, ecsv_path, run_cli
):
    """--event-time schedules a sky map that carries no trigger time."""
    result = run_cli(
        app,
        "schedule",
        skymap_without_gps_time,
        ecsv_path,
        "--mission=uvex",
        "--bandpass=NUV",
        "--nside=32",
        "--deadline=2hour",
        "--timelimit=10s",
        "--no-appmag-dist",
        "--event-time=2026-03-01T00:00:00",
    )
    assert result.exit_code == 0
    assert QTable.read(ecsv_path).meta["args"]["event_time"] == (
        "2026-03-01T00:00:00.000"
    )


def test_event_time_overrides_the_sky_map(fits_path, ecsv_path, run_cli):
    """An explicit trigger time takes precedence over the sky map header."""
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        "--mission=uvex",
        "--bandpass=NUV",
        "--nside=32",
        "--deadline=2hour",
        "--timelimit=10s",
        "--no-appmag-dist",
        "--event-time=2026-03-01T00:00:00",
    )
    assert result.exit_code == 0
    assert QTable.read(ecsv_path).meta["args"]["event_time"] == (
        "2026-03-01T00:00:00.000"
    )


def test_animate_uses_the_recorded_event_time(
    skymap_without_gps_time, ecsv_path, gif_path, run_cli
):
    """An animation needs no trigger time beyond the one the schedule records."""
    result = run_cli(
        app,
        "schedule",
        skymap_without_gps_time,
        ecsv_path,
        "--mission=uvex",
        "--bandpass=NUV",
        "--nside=32",
        "--deadline=4hour",
        "--timelimit=15s",
        "--no-appmag-dist",
        "--event-time=2026-03-01T00:00:00",
    )
    assert result.exit_code == 0
    result = run_cli(app, "animate", ecsv_path, gif_path, "--time-step=1hour")
    assert result.exit_code == 0
    assert gif_path.read_bytes().startswith(b"GIF89a")
