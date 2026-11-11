from typing import Annotated

import pytest
from astropy import units as u
from typer import Option, Typer

from ... import __version__, missions
from .. import core
from .._types import PhysicalQuantity


def test_version(run_cli):
    """Test the --version option."""
    result = run_cli(core.app, "--version")
    assert result.output.strip() == __version__


@pytest.mark.parametrize("default", ["100 s", 100 * u.s])
def test_quantity(run_cli, default):
    """A bare quantity option parses any unit; it pins no physical type."""

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

    # No physical type is pinned, so a mismatched unit is accepted.
    result, value = run("--foo=100meter")
    assert result.exit_code == 0
    assert value == 100 * u.meter

    result, _ = run("--foo=not-a-quantity")
    assert result.exit_code != 0
    assert "not-a-quantity" in result.output


def test_physical_quantity(run_cli):
    """An option typed with `PhysicalQuantity` is checked against that type."""
    TimeQuantity = PhysicalQuantity[u.physical.time]
    assert TimeQuantity is PhysicalQuantity[u.physical.time]  # cached
    result, value = _run_with_annotation(
        run_cli, Annotated[TimeQuantity, Option()], "--foo=200s"
    )
    assert result.exit_code == 0
    assert value == 200 * u.s

    result, _ = _run_with_annotation(
        run_cli, Annotated[TimeQuantity, Option()], "--foo=100meter"
    )
    assert result.exit_code != 0
    assert "100meter" in result.output
    assert "cannot be converted" in result.output
    assert "time" in result.output


def test_physical_quantity_list(run_cli):
    """A `PhysicalQuantity` works as a list element type, unlike an annotated one."""
    TimeQuantity = PhysicalQuantity[u.physical.time]
    result, value = _run_with_annotation(
        run_cli, Annotated[list[TimeQuantity], Option()], "--foo=1s", "--foo=2min"
    )
    assert result.exit_code == 0
    assert value == [1 * u.s, 2 * u.min]

    result, _ = _run_with_annotation(
        run_cli, Annotated[list[TimeQuantity], Option()], "--foo=100meter"
    )
    assert result.exit_code != 0
    assert "100meter" in result.output
    assert "cannot be converted" in result.output
    assert "time" in result.output


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

    # "Mission" is the base class itself, not an instance of it, and
    # `uvex_downlink_orientation` is a helper function; neither is a valid
    # choice even though both are in `missions.__all__`.
    result, value = run("--foo=Mission")
    assert result.exit_code != 0
    assert "'Mission' is not one of" in result.output

    result, value = run("--foo=uvex_downlink_orientation")
    assert result.exit_code != 0
    assert "'uvex_downlink_orientation'" in result.output
    assert "is not" in result.output
