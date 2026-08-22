"""
Common type aliases for :mod:`m4opt.models`.
"""

from typing import TYPE_CHECKING, TypeAlias, Union

import numpy as np
from astropy.units import Quantity, UnitBase
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from m4opt.models.core._parameters import Parameter


# =========================================================================== #
# UNITS                                                                       #
# =========================================================================== #

type UnitLike = str | UnitBase | None


# =========================================================================== #
# NUMERICAL INPUTS                                                            #
# =========================================================================== #

# Any unitless numerical input accepted by NumPy.
type NumericalInput = ArrayLike

# Any numerical input that may additionally carry physical units.
type PhysicalInput = ArrayLike | Quantity


# =========================================================================== #
# NORMALIZED NUMERICAL VALUES                                                  #
# =========================================================================== #

# Standard floating-point NumPy array used internally.
type FloatArray = NDArray[np.float64]

# Normalized scalar-or-array value used by numerical kernels.
type FloatValue = float | FloatArray

# Numerical result that may be scalar or array-valued.
type FloatResult = float | FloatArray


# =========================================================================== #
# MODEL PARAMETERS                                                            #
# =========================================================================== #

# User-facing physical parameter value.
type ParameterValue = PhysicalInput

# Unit-stripped CGS value accepted at public coercion boundaries.
type CGSParameterInput = ArrayLike

# Normalized unit-stripped CGS value used internally.
type CGSParameterValue = FloatValue

# Constructor override: replacement Parameter or fixed constant value.
#
# Deliberately a `typing.Union`, not a PEP 695 `type` alias: `Parameter` is
# only importable under TYPE_CHECKING (avoiding a circular import), and a
# PEP 695 alias evaluates its forward references lazily but *unquoted* --
# `"Parameter" | Quantity` would raise TypeError if ever forced (e.g. by
# `typing.get_type_hints`), since `Parameter` doesn't exist at runtime.
# `typing.Union` stores the quoted name as a `ForwardRef` instead, which
# tolerates never being resolved.
OverrideValue: TypeAlias = Union["Parameter", Quantity, float, int]  # noqa: UP040

# A concrete scalar used as a Parameter's `scale`, or to pin it via `fix()`.
# Unlike `ParameterValue`, this excludes bare arrays -- a scale or fixed
# value must be a single scalar.
type ScalarPhysicalValue = Quantity | float | int


# =========================================================================== #
# COMMON CONTAINERS                                                           #
# =========================================================================== #

type ParameterSamples = dict[str, Quantity | FloatArray]
type ParameterValues = dict[str, ParameterValue]
type CGSParameterValues = dict[str, CGSParameterValue]


# =========================================================================== #
# RANDOM NUMBER GENERATION                                                     #
# =========================================================================== #

type RNGInput = np.random.Generator | int | None
