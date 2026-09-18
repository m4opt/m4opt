from typing import Annotated

import pytest
from astropy import units as u
from typer import Option, Typer

from ... import __version__, missions
from .. import core


def test_version(run_cli):
    """Test the --version option."""
    result = run_cli(core.app, "--version")
    assert result.output.strip() == __version__


@pytest.mark.parametrize("default", ["100 s", 100 * u.s])
def test_quantity(run_cli, default):
    """Test CLI with quantity arguments."""

    def run(*args):
        app = Typer()
        value = None

        @app.command()
        def main(foo: u.Quantity = default):
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


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[u.Quantity[u.physical.time], Option()],
        Annotated[u.Quantity[u.s], Option()],
        Annotated[u.Quantity, u.physical.time, Option()],
    ],
    ids=["physical type", "unit", "metadata"],
)
def test_quantity_annotation(run_cli, annotation):
    """An option with no default is checked against the physical type it pins."""
    result, value = _run_with_annotation(run_cli, annotation, "--foo=200s")
    assert result.exit_code == 0
    assert value == 200 * u.s

    result, _ = _run_with_annotation(run_cli, annotation, "--foo=100meter")
    assert result.exit_code != 0
    assert "cannot be converted to time" in result.output


@pytest.mark.parametrize(
    "annotation",
    [
        Annotated[list[u.Quantity[u.physical.time]], Option()],
        Annotated[list[u.Quantity[u.s]], Option()],
        Annotated[list[u.Quantity], u.physical.time, Option()],
    ],
    ids=["physical type", "unit", "metadata"],
)
def test_quantity_list_annotation(run_cli, annotation):
    """Typer rejects an annotated list element, so the metadata is stripped."""
    result, value = _run_with_annotation(run_cli, annotation, "--foo=1s", "--foo=2min")
    assert result.exit_code == 0
    assert value == [1 * u.s, 2 * u.min]

    result, _ = _run_with_annotation(run_cli, annotation, "--foo=100meter")
    assert result.exit_code != 0
    assert "cannot be converted to time" in result.output


def _run_with_annotation(run_cli, annotation, *args):
    app = Typer()
    value = None

    @app.command()
    def main(foo: annotation):
        nonlocal value
        value = foo

    return run_cli(app, *args), value


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
