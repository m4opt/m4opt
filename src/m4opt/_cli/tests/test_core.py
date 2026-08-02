from astropy import units as u
from typer import Typer

from ... import __version__, missions
from .. import core


def test_version(run_cli):
    """Test the --version option."""
    result = run_cli(core.app, "--version")
    assert result.output.strip() == __version__


def test_quantity(run_cli):
    """Test CLI with quantity arguments."""

    def run(*args):
        app = Typer()
        value = None

        @app.command()
        def main(foo: u.Quantity = "100 s"):
            nonlocal value
            value = foo

        result = run_cli(app, *args)
        return result, value

    result, value = run()
    assert result.exit_code == 0
    assert value == 100 * u.s

    result, value = run("--foo=200s")
    assert result.exit_code == 0
    assert value == 200 * u.s

    result, value = run("--foo=100meter")
    assert result.exit_code != 0
    assert "value '100meter' cannot be" in result.output


def test_mission(run_cli):
    """Test CLI with mission arguments."""

    def run(*args):
        app = Typer()
        value = None

        @app.command()
        def main(foo: missions.Mission = missions.uvex):
            nonlocal value
            value = foo

        result = run_cli(app, *args)
        return result, value

    result, value = run()
    assert result.exit_code == 0
    assert value == missions.uvex

    result, value = run("--foo=ultrasat")
    assert result.exit_code == 0
    assert value == missions.ultrasat

    result, value = run("--foo=bar")
    assert result.exit_code != 0
    assert "'bar' is not one of" in result.output
