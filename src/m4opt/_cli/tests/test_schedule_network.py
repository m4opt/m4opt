from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable
from click import UsageError

from ... import networks
from .. import app
from ..schedule_network import (
    _per_mission_spec,
    load_config,
    prepare_visit_configuration,
    resolve_params,
    schedule_network,
)


def test_schedule_network_writes_a_schedule_for_each_mission(tmp_path, run_cli):
    config_path = tmp_path / "schedule.toml"
    config_path.write_text(
        '[default]\nskygrid = "sinusoidal"\nmax_fields = 1\n',
        encoding="utf-8",
    )
    schedule_path = tmp_path / "schedule"
    skymap_path = Path(__file__).parent / "data" / "800.fits"

    result = run_cli(
        app,
        "schedule-network",
        skymap_path,
        schedule_path,
        "--config",
        config_path,
        "--network=ground",
        "--nside=16",
        "--deadline=24hour",
        "--time-step=1hour",
        "--timelimit=15s",
        "--exptime-min=60s",
        "--no-appmag-dist",
    )

    assert result.exit_code == 0
    for name in ("ztf", "rubin"):
        table = QTable.read(tmp_path / f"schedule_{name}.ecsv")
        assert len(table) == 1
        assert table["action"][0] == "observe"
        assert table.meta["args"]["mission"] == name


def test_schedule_network_is_registered(run_cli):
    result = run_cli(app, "schedule-network", "--help")

    assert result.exit_code == 0
    assert "--network" in result.output
    assert "ground" in result.output


def test_per_mission_config_overrides_defaults(tmp_path):
    config_path = tmp_path / "schedule.toml"
    config_path.write_text(
        '[default]\nmax_fields = 3\n[rubin]\nmax_fields = 2\n',
        encoding="utf-8",
    )
    spec = _per_mission_spec(schedule_network)
    config = load_config(config_path, networks.ground, spec)
    cli_values = {
        "exptime_min": [30 * u.s],
        "skygrid": None,
        "delay": 0 * u.day,
        "deadline": 1 * u.day,
        "exptime_max": np.inf * u.s,
        "bandpass": None,
        "max_fields": 50,
    }

    params = resolve_params(networks.tarot, config, spec, cli_values, visits=1)

    assert params["rubin"].max_fields == 2
    assert params["ztf"].max_fields == 3


def test_visit_configuration_cycles_bandpasses_and_exposure_times():
    bands, exposure_times = prepare_visit_configuration(
        visits=3,
        bandpass=["g", "r"],
        exptime_min=[30 * u.s, 60 * u.s],
    )

    assert bands == ["g", "r", "g"]
    np.testing.assert_array_equal(exposure_times, [30, 60, 30])


@pytest.mark.parametrize(
    "visits, bandpass, exptime_min, message",
    [
        (0, ["g"], [30 * u.s], "at least one"),
        (1, ["g"], [], "minimum exposure time"),
    ],
)
def test_visit_configuration_rejects_invalid_input(
    visits, bandpass, exptime_min, message
):
    with pytest.raises(UsageError, match=message):
        prepare_visit_configuration(visits, bandpass, exptime_min)
