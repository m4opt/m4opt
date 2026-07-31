"""Spacecraft dynamics functions.

Notes
-----
Most of the functions in this module will live somewhere else eventually when
this package is more organized.
"""

from ._roll import nominal_roll
from ._slew import (
    AngularMotionProfile,
    AltAzSlew,
    EigenAxisSlew,
    EquatorialSlew,
    GroundSlew,
    MixedCoordSlew,
    Slew,
    SlewComponent,
)

__all__ = (
    "AngularMotionProfile",
    "EigenAxisSlew",
    "GroundComponentSlew",
    "GroundSlew",
    "Slew",
    "nominal_roll",
)
