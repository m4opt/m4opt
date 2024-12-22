from astropy import units as u
from astropy.table import QTable
from astropy.utils.data import download_file

from .. import app


def test_end_to_end(tmp_path, run_cli):
    """Exercise the scheduler with a trivially short time limit."""
    url = "https://gracedb.ligo.org/api/superevents/S190425z/files/bayestar.fits"
    fits_path = download_file(url, cache=True)
    ecsv_path = tmp_path / "example.ecsv"
    gif_path = tmp_path / "example.gif"

    result = run_cli(app, "schedule", fits_path, str(ecsv_path), "--timelimit=1s")
    assert result.exit_code == 0
    table = QTable.read(ecsv_path)
    assert len(table) == 0
    assert table.meta["solution_status"] == "time limit exceeded"
    assert table.meta["objective_value"] == 0
    assert table.meta["total_time"]["slack"] == 1 * u.day

    result = run_cli(app, "animate", str(ecsv_path), str(gif_path))
    assert result.exit_code == 0
    assert gif_path.read_bytes().startswith(b"GIF89a")