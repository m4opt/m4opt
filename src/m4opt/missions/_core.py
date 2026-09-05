from collections.abc import Hashable
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
from numpy import typing as npt
from regions import Region, Regions

from ..constraints import Constraint
from ..dynamics import Slew
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

    slew: Slew
    """Slew time model."""

    skygrid: SkyCoord | dict[Hashable, SkyCoord]
    """Grid of reference pointings.
    May be either a single SkyCoord instance or a dictionary of named SkyCoord grids
    (e.g., for different survey strategies such as "allsky" or "non-overlap").
    """

    detector: Detector | None = None
    """Detector model."""

    field_ids: npt.NDArray | dict[Hashable, npt.NDArray] | None = None
    """Identifiers the mission gives its reference pointings, if it names them.

    Parallel to :attr:`skygrid`, and a dictionary with the same keys when
    :attr:`skygrid` is one. These are the mission's own identifiers, which need
    not be contiguous or ordered: ZTF's run from 1 to 1897 with gaps across
    1778 fields.
    """
