from typing import Annotated

from astropy import units as u
from typer import Option, Typer

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


def test_quantity_physical_type_from_annotation(run_cli):
    """An option with no default is checked against the physical type it pins."""

    def run(annotation, *args):
        app = Typer()
        value = None

        @app.command()
        def main(foo: annotation):
            nonlocal value
            value = foo

        return run_cli(app, *args), value

    # A required option has no default to take a physical type from.
    required = Annotated[u.Quantity, u.physical.time, Option()]
    result, value = run(required, "--foo=200s")
    assert result.exit_code == 0
    assert value == 200 * u.s

    result, _ = run(required, "--foo=100meter")
    assert result.exit_code != 0
    assert "cannot be converted to time" in result.output

    # Repeated and optional forms pin it the same way.
    repeated = Annotated[list[u.Quantity], u.physical.time, Option()]
    result, value = run(repeated, "--foo=1s", "--foo=2min")
    assert result.exit_code == 0
    assert value == [1 * u.s, 2 * u.min]

    result, _ = run(repeated, "--foo=1s", "--foo=100meter")
    assert result.exit_code != 0
    assert "cannot be converted to time" in result.output


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
