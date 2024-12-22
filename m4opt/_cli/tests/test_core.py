from astropy import units as u
from typer import Typer
from typer.testing import CliRunner

from .. import core  # noqa: F401

runner = CliRunner()


def test_quantity():
    """Test CLI with quantity arguments."""

    def run(*args):
        app = Typer()
        value = None

        @app.command()
        def main(foo: u.Quantity = "100 s"):
            nonlocal value
            value = foo

        result = runner.invoke(app, args)
        return result, value

    result, value = run()
    assert result.exit_code == 0
    assert value == 100 * u.s

    result, value = run("--foo=200s")
    assert result.exit_code == 0
    assert value == 200 * u.s

    result, value = run("--foo=100meter")
    assert result.exit_code != 0
    assert (
        "Invalid value for '--foo': value '100meter' cannot be converted to time"
        in result.output
    )
