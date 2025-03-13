"""Spacecraft dynamics functions.

Notes
-----
Most of the functions in this module will live somewhere else eventually when
this package is more organized.
"""

from ._roll import nominal_roll
from ._slew import EigenAxisSlew, Slew

__all__ = ("nominal_roll", "EigenAxisSlew", "Slew")
