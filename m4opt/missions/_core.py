from collections.abc import Collection
from dataclasses import dataclass

from regions import Region, Regions

from ..constraints import Constraint
from ..models import Detector
from ..orbit import Orbit


@dataclass
class Mission:
    """Base class for all missions."""

    fov: Region | Regions
    """Instrument field of view."""

    constraints: Collection[Constraint]
    """Field of regard constraints."""

    detector: Detector
    """Detector mdoel."""

    orbit: Orbit
    """Orbit of spacecraft."""
