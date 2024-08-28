from ._astropy_init import *  # noqa
from ._roll import nominal_roll
from ._slew import slew_separation, slew_time

__all__ = ("nominal_roll", "slew_separation", "slew_time")
