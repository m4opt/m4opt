"""
Common type aliases for :mod:`m4opt.models`.
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from astropy.units import Quantity, UnitBase
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from m4opt.models.core._parameters import Parameter


# =========================================================================== #
# UNITS                                                                       #
# =========================================================================== #

UnitLike: type = str | UnitBase | None


# =========================================================================== #
# NUMERICAL INPUTS                                                            #
# =========================================================================== #

# Any unitless numerical input accepted by NumPy.
NumericalInput: type = ArrayLike

# Any numerical input that may additionally carry physical units.
PhysicalInput: type = ArrayLike | Quantity


# =========================================================================== #
# NORMALIZED NUMERICAL VALUES                                                  #
# =========================================================================== #

# Standard floating-point NumPy array used internally.
FloatArray: type = NDArray[np.float64]

# Normalized scalar-or-array value used by numerical kernels.
FloatValue: type = float | FloatArray

# Numerical result that may be scalar or array-valued.
FloatResult: type = float | FloatArray


# =========================================================================== #
# MODEL PARAMETERS                                                            #
# =========================================================================== #

# User-facing physical parameter value.
ParameterValue: type = PhysicalInput

# Unit-stripped CGS value accepted at public coercion boundaries.
CGSParameterInput: type = ArrayLike

# Normalized unit-stripped CGS value used internally.
CGSParameterValue: type = FloatValue

# Constructor override: replacement Parameter or fixed constant value.
OverrideValue: type = Union["Parameter", Quantity, float, int]


# =========================================================================== #
# COMMON CONTAINERS                                                           #
# =========================================================================== #

ParameterSamples: type = dict[str, Quantity | FloatArray]
ParameterValues: type = dict[str, ParameterValue]
CGSParameterValues: type = dict[str, CGSParameterValue]


# =========================================================================== #
# RANDOM NUMBER GENERATION                                                     #
# =========================================================================== #

RNGInput: type = np.random.Generator | int | None
