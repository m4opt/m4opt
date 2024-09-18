"""Field of regard constraints.

The field of regard is the region of the sky that a detector is allowed to
point at. These classes model various constraints on the field of regard, as
functions of the location of the detector in space (an
:class:`~astropy.coordinates.EarthLocation` instance), the target at which the
detector is pointed (a :class:`~astropy.coordinates.SkyCoord` instance), and
the time of the observation (a :class:`~astropy.time.Time` instance).
"""

from ._body_separation import MoonSeparationConstraint, SunSeparationConstraint
from ._core import Constraint
from ._earth_limb import EarthLimbConstraint

__all__ = (
    "Constraint",
    "EarthLimbConstraint",
    "MoonSeparationConstraint",
    "SunSeparationConstraint",
)
