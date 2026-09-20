from typing import Any

import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import (
    array_shapes,
    arrays,
    mutually_broadcastable_shapes,
    scalar_dtypes,
)
from hypothesis.strategies import DataObject, DrawFn, composite, data, integers

from .._short_circuit import (
    BoolGenericFunction,
    BoolNDArray,
    logical_and_short_circuit,
    logical_or_short_circuit,
)


def logical_and_short_circuit_slow(
    lhs: BoolNDArray, rhs_func: BoolGenericFunction, *args: np.ndarray
) -> BoolNDArray:
    return lhs & rhs_func(*args)


def logical_or_short_circuit_slow(
    lhs: BoolNDArray, rhs_func: BoolGenericFunction, *args: np.ndarray
) -> BoolNDArray:
    return lhs | rhs_func(*args)


def stable_hash(arg):
    """
    A version of :func:`hash` that is stable for NaN values.

    The builtin :func:`hash` function does not return a stable value for NaN.
    """
    try:
        isnan = np.isnan(arg)
    except TypeError:
        pass
    else:
        if isnan:
            arg = "NaN"
    return hash(arg)


def rhs_func_scalar(*args: Any) -> bool:
    """An arbitrary boolean-returning pure function of the arguments."""
    return bool(hash(tuple(stable_hash(arg) for arg in args)) & 1)


@composite
def compatible_args(draw: DrawFn, lhs: np.ndarray):
    nargs = draw(integers(min_value=1, max_value=5))
    shapes = draw(mutually_broadcastable_shapes(num_shapes=nargs, base_shape=lhs.shape))
    return [
        draw(arrays(dtype=scalar_dtypes(), shape=shape))
        for shape in shapes.input_shapes
    ]


@given(lhs=arrays(bool, array_shapes(min_dims=0)), data=data())
def test_logical_short_circuit(lhs: np.ndarray, data: DataObject):
    args = data.draw(compatible_args(lhs))
    rhs_func = np.vectorize(rhs_func_scalar)

    np.testing.assert_array_equal(
        logical_and_short_circuit(lhs, rhs_func, *args),
        logical_and_short_circuit_slow(lhs, rhs_func, *args),
    )

    np.testing.assert_array_equal(
        logical_or_short_circuit(lhs, rhs_func, *args),
        logical_or_short_circuit_slow(lhs, rhs_func, *args),
    )
