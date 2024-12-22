from astropy import units as u
from typer import Typer
from typer.testing import CliRunner

from .. import core  # noqa: F401


def test_quantity():
    """Test CLI with quantity arguments."""
    app = Typer()
    runner = CliRunner()

    @app.command()
    def main(foo: u.Quantity = "100 s"):
        assert foo == 100 * u.s

    result = runner.invoke(app)
    assert result.exit_code == 0

    result = runner.invoke(app, ["--foo=100s"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["--foo=100meter"])
    assert result.exit_code != 0
    assert (
        "Invalid value for '--foo': value '100meter' cannot be converted to time"
        in result.output
    )
