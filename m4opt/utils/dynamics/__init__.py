"""Spacecraft dynamics functions.

Notes
-----
Most of the functions in this module will live somewhere else eventually when
this package is more organized.
"""

from ._roll import nominal_roll
from ._slew import slew_separation, slew_time

__all__ = ("nominal_roll", "slew_separation", "slew_time")
