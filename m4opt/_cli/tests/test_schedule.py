from astropy import units as u
from astropy.table import QTable
from astropy.utils.data import download_file

from .. import app


def test_schedule(tmp_path, run_cli):
    """Exercise the scheduler with a trivially short time limit."""
    url = "https://gracedb.ligo.org/api/superevents/S190425z/files/bayestar.fits"
    in_path = download_file(url, cache=True)
    out_path = tmp_path / "example.ecsv"
    result = run_cli(app, "schedule", in_path, str(out_path), "--timelimit=1s")
    assert result.exit_code == 0
    table = QTable.read(out_path)
    assert len(table) == 0
    assert table.meta["solution_status"] == "time limit exceeded"
    assert table.meta["objective_value"] == 0
    assert table.meta["total_time"]["slack"] == 1 * u.day
