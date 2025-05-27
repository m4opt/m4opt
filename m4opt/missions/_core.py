from collections.abc import Collection, Hashable
from dataclasses import dataclass

from astropy.coordinates import SkyCoord
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

    constraints: Collection[Constraint]
    """Field of regard constraints."""

    detector: Detector
    """Detector model."""

    observer_location: ObserverLocation
    """Orbit of spacecraft."""

    slew: Slew
    """Slew time model."""

    skygrid: SkyCoord | dict[Hashable, SkyCoord]
    """Grid of reference pointings.
    May be either a single SkyCoord instance or a dictionary of named SkyCoord grids
    (e.g., for different survey strategies such as "allsky" or "non-overlap").
    """
