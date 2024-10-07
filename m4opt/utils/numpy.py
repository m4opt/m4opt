"""Utilities for working with Numpy arrays."""

import numpy as np
from astropy import units as u


def arange_with_units(*args: u.Quantity) -> u.Quantity:
    """A version of :obj:`numpy.arange` that works with Astropy units.

    Examples
    --------
    >>> from astropy import units as u
    >>> from m4opt.utils.numpy import arange_with_units
    >>> arange_with_units(0 * u.m, 10 * u.m)
    <Quantity [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.] m>

    Notes
    -----
    :obj:`numpy.arange` is known not to work with Astropy units. See:

    - https://docs.astropy.org/en/stable/known_issues.html#numpy-array-creation-functions-cannot-be-used-to-initialize-quantity
    - https://github.com/astropy/astropy/issues/17001
    - https://github.com/astropy/astropy/pull/17120
    """
    # FIXME: Remove this when we depend upon astropy >= 7.0.0.
    first, *rest = args
    unit = first.unit
    return np.arange(first.value, *(arg.to_value(unit) for arg in rest)) * unit


def atmost_1d(a):
    """Force an array-like object to have no more than 1 dimension.

    Examples
    --------
    >>> from m4opt.utils.numpy import atmost_1d
    >>> atmost_1d([[1, 2], [3, 4]])
    array([1, 2, 3, 4])
    >>> atmost_1d([1, 2, 3, 4])
    array([1, 2, 3, 4])
    >>> atmost_1d([])
    array([], dtype=float64)
    >>> atmost_1d(1)
    1

    See also
    --------
    numpy.atleast_1d
    """
    if isinstance(a, (np.ndarray, list, tuple)):
        a = np.ravel(a)
    return a


def clump_nonzero(a):
    """Find intervals of nonzero values in an array, row by row.

    Examples
    --------
    >>> from m4opt.utils.numpy import clump_nonzero
    >>> clump_nonzero([[0, 0, 1, 1, 0, 0]])
    [array([[2, 4]])]
    >>> clump_nonzero([[1, 1, 0, 0, 1, 1]])
    [array([[0, 2],
           [4, 6]])]
    >>> clump_nonzero([[1, 1, 1, 1, 1, 1]])
    [array([[0, 6]])]
    >>> clump_nonzero([[0, 0, 0, 0, 0, 0]])
    [array([], shape=(0, 2), dtype=int64)]

    See also
    --------
    numpy.ma.clump_masked, numpy.ma.clump_unmasked
    """
    # FIXME: see https://github.com/numpy/numpy/issues/27374
    masked_array = np.ma.array(np.empty_like(a, dtype=np.void), mask=a)
    return [
        np.asarray(
            [(slice.start, slice.stop) for slice in np.ma.clump_masked(row)],
            dtype=np.intp,
        ).reshape((-1, 2))
        for row in masked_array
    ]


def full_indices(n):
    """Calculate the indices of all of the elements of a square array.

    Examples
    --------
    >>> from m4opt.utils.numpy import full_indices
    >>> full_indices(2)
    [array([0, 0, 1, 1]), array([0, 1, 0, 1])]
    >>> full_indices(0)
    [array([], dtype=int64), array([], dtype=int64)]

    See also
    --------
    numpy.tril_indices, numpy.triu_indices
    """
    return [x.ravel() for x in np.mgrid[:n, :n]]
