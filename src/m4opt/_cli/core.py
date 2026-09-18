import typing
from typing import Annotated

import typer
from astropy import units as u
from astropy.time import Time
from typer._click.types import ParamType
from typer._types import TyperChoice
from typer.main import get_click_type as _get_click_type
from typer.main import lenient_issubclass

from .. import __version__, missions

app = typer.Typer(pretty_exceptions_show_locals=False)


def version_callback(value: bool):
    if value:
        print(__version__)
        raise typer.Exit()


class QuantityClickType(ParamType):
    name = "Astropy quantity"

    def __init__(self, physical_type=None):
        self.physical_type = physical_type

    def convert(self, value, param, ctx):
        result = u.Quantity(value)
        target_physical_type = self.physical_type
        if (
            target_physical_type is None
            and param is not None
            and (default := param.default) is not None
        ):
            # An option that pins no physical type is held to its default's.
            target_physical_type = u.get_physical_type(u.Quantity(default))
        if (
            target_physical_type is not None
            and u.get_physical_type(result) != target_physical_type
        ):
            self.fail(
                f"value '{value}' cannot be converted to {target_physical_type}",
                param,
                ctx,
            )
        return result


class TimeClickType(ParamType):
    name = "Astropy time"

    def convert(self, value, param, ctx):
        if isinstance(value, Time):
            return value
        try:
            return Time(value)
        except ValueError:
            self.fail(f"value '{value}' is not a recognized time", param, ctx)


class MissionClickType(TyperChoice):
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
    elif lenient_issubclass(annotation, Time):
        return TimeClickType()
    elif lenient_issubclass(annotation, missions.Mission):
        return MissionClickType()
    else:
        return _get_click_type(annotation=annotation, parameter_info=parameter_info)


typer.main.get_click_type = get_click_type


def _physical_type(annotation):
    """
    The physical type an annotation pins, if it pins one.

    Typer discards the metadata of an :obj:`~typing.Annotated` annotation before
    it builds a parameter, so the physical type has to be recovered from the
    function's own type hints.
    """
    if isinstance(annotation, u.PhysicalType):
        return annotation
    for arg in typing.get_args(annotation):
        if (found := _physical_type(arg)) is not None:
            return found
    return None


_get_params_from_function = (
    typer.main.get_params_convertors_ctx_param_name_from_function
)


def get_params_convertors_ctx_param_name_from_function(callback):
    """
    Monkeypatch for Typer to check a quantity against its annotation.

    An option annotated ``u.Quantity[u.physical.time]``, or one whose
    :obj:`~typing.Annotated` metadata names a physical type, is checked against
    that rather than against the physical type of its default value. A required
    option has no default to check against.
    """
    params, converters, ctx_name = _get_params_from_function(callback)
    if callback is not None:
        hints = typing.get_type_hints(callback, include_extras=True)
        for param in params:
            physical_type = _physical_type(hints.get(param.name))
            if physical_type is not None and isinstance(param.type, QuantityClickType):
                param.type = QuantityClickType(physical_type)
    return params, converters, ctx_name


typer.main.get_params_convertors_ctx_param_name_from_function = (
    get_params_convertors_ctx_param_name_from_function
)


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
