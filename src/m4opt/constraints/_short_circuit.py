from typing import Any, Protocol

import numpy as np

type BoolNDArray = np.ndarray[tuple[Any, ...], np.dtype[np.bool]]


class BoolGenericFunction(Protocol):
    def __call__(self, *args: np.ndarray) -> BoolNDArray: ...


def _logical_short_circuit(
    decisive: bool,
    combine: np.ufunc,
    lhs: BoolNDArray,
    rhs_func: BoolGenericFunction,
    args: tuple[np.ndarray, ...],
) -> BoolNDArray:
    """
    Shared implementation of `logical_and_short_circuit` and `logical_or_short_circuit`.

    `decisive` is the value of `lhs` that alone determines the result of
    `combine`, without needing to evaluate `rhs_func`: `False` for `&`,
    `True` for `|`.
    """
    shape = np.broadcast_shapes(lhs.shape, *(arg.shape for arg in args))
    ndim = len(shape)
    lhs = lhs.reshape((1,) * (ndim - lhs.ndim) + lhs.shape)

    undecided = np.logical_xor(lhs, decisive)
    if not undecided.any():
        return np.broadcast_to(lhs, shape).copy()[()]

    # For each axis, find the indices along that axis for which `lhs` is
    # undecided for at least one value of the other axes. Axes on which
    # `lhs` is itself broadcast (size 1) cannot be narrowed this way, and
    # are left alone. Restricting each argument to only these indices, on
    # only the axes that the argument does not itself broadcast over,
    # shrinks the arrays passed to `rhs_func` without ever forming a full,
    # broadcast array for `lhs` or any argument.
    undecided_indices = [
        None
        if undecided.shape[axis] == 1
        else np.flatnonzero(
            undecided.any(axis=tuple(a for a in range(ndim) if a != axis))
        )
        for axis in range(ndim)
    ]

    def compress(arr: np.ndarray) -> np.ndarray:
        arr = arr.reshape((1,) * (ndim - arr.ndim) + arr.shape)
        for axis, indices in enumerate(undecided_indices):
            if indices is not None and arr.shape[axis] != 1:
                arr = np.take(arr, indices, axis=axis)
        return arr

    reduced_result = combine(compress(lhs), rhs_func(*(compress(arg) for arg in args)))

    result = np.broadcast_to(lhs, shape).copy()
    result[
        np.ix_(
            *(
                np.arange(shape[axis]) if indices is None else indices
                for axis, indices in enumerate(undecided_indices)
            )
        )
    ] = reduced_result
    return result[()]


def logical_and_short_circuit(
    lhs: BoolNDArray, rhs_func: BoolGenericFunction, *args: np.ndarray
) -> BoolNDArray:
    """
    Compute logical and on Numpy arrays employing short circuit evaluation.

    The function call `logical_and_short_circuit(lhs, rhs_func, args)` is
    equivalent to, but potentially faster than, `lhs & rhs_func(*args)`.

    Materialization of broadast arrays is delayed as late as possible.

    The argument `rhs_func` may be any pure, elementwise function.
    """
    return _logical_short_circuit(False, np.logical_and, lhs, rhs_func, args)


def logical_or_short_circuit(
    lhs: BoolNDArray, rhs_func: BoolGenericFunction, *args: np.ndarray
) -> BoolNDArray:
    """
    Compute logical or on Numpy arrays employing short circuit evaluation.

    The function call `logical_or_short_circuit(lhs, rhs_func, args)` is
    equivalent to, but potentially faster than, `lhs | rhs_func(*args)`.

    Materialization of broadast arrays is delayed as late as possible.

    The argument `rhs_func` may be any pure, elementwise function.
    """
    return _logical_short_circuit(True, np.logical_or, lhs, rhs_func, args)
