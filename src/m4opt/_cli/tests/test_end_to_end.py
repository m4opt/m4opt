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
            observations["duration"] + 1e-3 * u.s
            >= u.Quantity(table.meta["args"]["exptime_min"]).min()
        ).all()
        assert (observations["duration"] <= table.meta["args"]["exptime_max"]).all()

        grid = getattr(missions, table.meta["args"]["mission"]).skygrid
        if isinstance(grid, dict):
            grid = grid[table.meta["args"]["skygrid"]]
        field_ids = observations["field_id"]
        assert not np.any(np.ma.getmaskarray(field_ids)), (
            "an observation names the field it points at"
        )
        # The identifier indexes the grid, so no translation is needed.
        separation = grid[np.asarray(field_ids)].separation(
            observations["target_coord"]
        )
        np.testing.assert_allclose(np.asarray(separation.deg), 0, atol=1e-9)
        assert np.all(
            np.ma.getmaskarray(table[table["action"] == "slew"]["field_id"])
        ), "a slew points at no field"

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


def test_fixed_exptime_with_appmag_dist(fits_path, ecsv_path, run_cli, mission_args):
    """
    Fixed exposure time mode should work when appmag_dist is True (default).

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
        "--exptime-min=300s",
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
        run_cli(
            app,
            "schedule",
            skymap_without_gps_time,
            ecsv_path,
            "--mission=uvex",
            "--exptime-min=300s",
        )


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
        "--exptime-min=300s",
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
        "--exptime-min=300s",
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
        "--exptime-min=300s",
    )
    assert result.exit_code == 0
    result = run_cli(app, "animate", ecsv_path, gif_path, "--time-step=1hour")
    assert result.exit_code == 0
    assert gif_path.read_bytes().startswith(b"GIF89a")


@pytest.mark.parametrize(
    "exptime_args",
    [("--exptime-min=300s",), ("--exptime-min=120s", "--exptime-min=300s")],
    ids=["one exposure time", "one per bandpass"],
)
def test_a_field_observable_in_several_windows(
    fits_path, ecsv_path, run_cli, exptime_args
):
    """A field that rises and sets several times still schedules.

    A ground based telescope sees a field in a separate window each night, and
    the exposure times run along the visits rather than along those windows, so
    the two have to be broadcast against each other rather than zipped.
    """
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        "--mission=ztf",
        "--bandpass=g",
        "--bandpass=r",
        "--visits=2",
        "--nside=16",
        "--max-fields=6",
        # Long enough that each field sets and rises again more than twice.
        "--deadline=96hour",
        "--timelimit=20s",
        "--no-appmag-dist",
        *exptime_args,
    )
    assert result.exit_code == 0

    table = QTable.read(ecsv_path)
    observations = table[table["action"] == "observe"]
    if len(observations) == 0:
        pytest.skip("no fields were observable")
    observations.sort("start_time")

    # Each observation lies inside a window, and they do not overlap.
    starts = observations["start_time"].gps
    ends = starts + observations["duration"].to_value(u.s)
    assert (starts[1:] >= ends[:-1]).all()


def test_each_gap_between_visits_may_have_its_own_cadence(
    fits_path, ecsv_path, run_cli
):
    """Repeating --cadence separates each pair of consecutive visits in turn."""
    result = run_cli(
        app,
        "schedule",
        fits_path,
        ecsv_path,
        "--mission=ztf",
        "--bandpass=g",
        "--bandpass=r",
        "--visits=3",
        "--nside=16",
        "--max-fields=6",
        "--deadline=8hour",
        "--timelimit=30s",
        "--no-appmag-dist",
        "--exptime-min=300s",
        "--cadence=1min",
        "--cadence=45min",
    )
    assert result.exit_code == 0

    table = QTable.read(ecsv_path)
    observations = table[table["action"] == "observe"]
    if len(observations) == 0:
        pytest.skip("no fields were observable")

    for coord in {str(coord) for coord in observations["target_coord"]}:
        visits = observations[
            [str(each) == coord for each in observations["target_coord"]]
        ]
        visits.sort("start_time")
        starts = visits["start_time"].gps
        ends = starts + visits["duration"].to_value(u.s)
        gaps = starts[1:] - ends[:-1]
        assert gaps[0] >= 60 - 1e-3
        assert gaps[1] >= 45 * 60 - 1e-3
