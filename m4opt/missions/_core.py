from collections.abc import Collection
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
    """Instrument field of view."""

    constraints: Collection[Constraint]
    """Field of regard constraints."""

    detector: Detector
    """Detector model."""

    observer_location: ObserverLocation
    """Orbit of spacecraft."""

    slew: Slew
    """Slew time model."""

    skygrid: SkyCoord
    """Grid of reference pointings."""
