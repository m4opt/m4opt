from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from regions import Region, Regions

from ..constraints import Constraint
from ..dynamics import GroundSlew, Slew
from ..observer import ObserverLocation
from ..synphot import Detector


@dataclass(repr=False)
class Mission:
    """Base class for all missions."""

    name: str
    """Name of the mission."""

    fov: Region | Regions
    """Instrument field of view.

    The region is expected to represent the field of view of the instrument at
    a standard orientation of R.A.=0, Dec.=0, P.A.=0.

    See Also
    --------
    m4opt.fov.footprint, m4opt.fov.footprint_healpix
    """

    constraints: Constraint
    """Field of regard constraint.

    To add multiple constraints, combine them using boolean operations
    (``lhs & rhs``, ``lhs | rhs``, and ``~lhs``).

    See Also
    --------
    m4opt.constraints.LogicalAndConstraint,
    m4opt.constraints.LogicalOrConstraint,
    m4opt.constraints.LogicalNotConstraint
    """

    observer_location: ObserverLocation
    """Orbit of spacecraft."""

    slew: Slew | GroundSlew
    """Slew time model."""

    skygrid: SkyCoord | dict[Hashable, SkyCoord]
    """Grid of reference pointings.
    May be either a single SkyCoord instance or a dictionary of named SkyCoord grids
    (e.g., for different survey strategies such as "allsky" or "non-overlap").
    """

    detector: Detector | None = None
    """Detector model."""

    filter_exchange_time: u.Quantity[u.physical.time] = 0 * u.s
    """Time to exchange one bandpass filter for another."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Let Pydantic and Typer parse a mission by name."""
        return core_schema.no_info_plain_validator_function(cls._cli_parse)

    @classmethod
    def _cli_parse(cls, value: Any) -> "Mission":
        if isinstance(value, cls):
            return value
        # Deferred to avoid a circular import.
        from .. import missions

        # Only names bound to an actual mission are valid choices -- this
        # excludes `Mission` itself (the base class, not an instance of it)
        # and anything else in `missions.__all__` that isn't a mission, such
        # as the `uvex_downlink_orientation` helper function.
        choices = {
            name: obj
            for name in missions.__all__
            if isinstance(obj := getattr(missions, name), cls)
        }
        try:
            return choices[value]
        except KeyError:
            raise ValueError(
                f"{value!r} is not one of {', '.join(map(repr, choices))}."
            ) from None
