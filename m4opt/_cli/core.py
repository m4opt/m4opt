from typing import Annotated

import click
import typer
from astropy import units as u
from typer.main import get_click_type as _get_click_type
from typer.main import lenient_issubclass

from .. import __version__, missions

app = typer.Typer(pretty_exceptions_show_locals=False)


def version_callback(value: bool):
    if value:
        print(__version__)
        raise typer.Exit()


class QuantityClickType(click.ParamType):
    name = "Astropy quantity"

    def convert(self, value, param, ctx):
        result = u.Quantity(value)
        if param is not None and (default := param.default) is not None:
            target_physical_type = u.get_physical_type(u.Quantity(default))
            physical_type = u.get_physical_type(result)
            if physical_type != target_physical_type:
                self.fail(
                    f"value '{value}' cannot be converted to {target_physical_type}",
                    param,
                    ctx,
                )
        return result


class MissionClickType(click.Choice):
    def __init__(self):
        choices = [name for name in missions.__all__ if name[0].islower()]
        super().__init__(choices)

    def convert(self, value, *args, **kwargs):
        if isinstance(value, missions.Mission):
            return value
        else:
            return getattr(missions, super().convert(value, *args, **kwargs))


def get_click_type(*, annotation, parameter_info):
    """Monkeypatch for typer.main.get_click_type to add support for new types."""
    if lenient_issubclass(annotation, u.Quantity):
        return QuantityClickType()
    elif lenient_issubclass(annotation, missions.Mission):
        return MissionClickType()
    else:
        return _get_click_type(annotation=annotation, parameter_info=parameter_info)


typer.main.get_click_type = get_click_type


@app.callback()
def version(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print version and exit.",
        ),
    ] = False,
):
    pass
