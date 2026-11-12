"""
Teach Pydantic and Typer to parse Astropy quantities and times.

:class:`astropy.units.Quantity` and :class:`astropy.time.Time` are third-party
types. We can't add that method to directly in their own class bodies, so it is
set here instead.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from astropy import units as u
from astropy.time import Time
from pydantic_core import core_schema


def pydantic_parser(
    target: type,
) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
    """Patch a third-party class with a Pydantic validator."""

    def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def get_core_schema(cls: type, source_type: Any, handler: Any) -> Any:
            return core_schema.no_info_plain_validator_function(func)

        target.__get_pydantic_core_schema__ = classmethod(get_core_schema)  # type: ignore[attr-defined]
        return func

    return decorator


@pydantic_parser(u.Quantity)
def _parse_quantity(value: Any) -> u.Quantity:
    try:
        return u.Quantity(value)
    except TypeError as exc:
        # astropy raises TypeError for values it can't parse at all, but
        # Pydantic only treats ValueError (and AssertionError) as a reportable
        # validation failure, not TypeError.
        raise ValueError(str(exc)) from exc


@functools.cache
def _physical_quantity(physical_type: u.PhysicalType) -> type[u.Quantity]:
    class _Quantity(u.Quantity):
        pass

    @pydantic_parser(_Quantity)
    def _parse(value: Any) -> u.Quantity:
        result = _parse_quantity(value)
        if u.get_physical_type(result) != physical_type:
            raise ValueError(f"'{value}' cannot be converted to {physical_type}")
        return result

    _Quantity.__name__ = _Quantity.__qualname__ = f"PhysicalQuantity[{physical_type}]"
    return _Quantity


# FIXME: remove once the following TODO item in
# :meth:`astropy.units.Quantity.__class_getitem__` is addressed:
#
#   Quantity does not (yet) properly extend the NumPy generics types,
#   introduced in numpy v1.22+, instead just including the unit info as
#   metadata using Annotated.
#   TODO: ensure we do interact with NDArray.__class_getitem__.


if TYPE_CHECKING:
    # mypy only treats a subscripted class as a type if it is Generic, which
    # PhysicalQuantity -- deliberately -- is not: its subscript is a physical
    # type value, not a type parameter. So, for type checking only, it is a
    # plain alias for Quantity, which is itself subscriptable this way.
    PhysicalQuantity = u.Quantity
else:

    class PhysicalQuantity:
        """
        A `~astropy.units.Quantity` subtype restricted to one physical type.

        Subscript it with the desired physical type, e.g.
        ``PhysicalQuantity[u.physical.time]``, the same way you would
        subscript any other generic type such as ``list[int]``.

        Unlike ``u.Quantity[physical_type]``, this returns a genuine class,
        not an :obj:`~typing.Annotated` alias.
        """

        def __class_getitem__(cls, physical_type: u.PhysicalType) -> type[u.Quantity]:
            return _physical_quantity(physical_type)


@pydantic_parser(Time)
def _parse_time(value: Any) -> Time:
    if isinstance(value, Time):
        return value
    try:
        return Time(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a recognized time") from None
