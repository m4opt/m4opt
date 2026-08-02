"""Utilities for working with Numpy arrays."""

import numpy as np


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


def clump_nonzero_inclusive(a):
    """Like clump_nonzero, but return closed rather than half-open intervals.

    Examples
    --------
    >>> from m4opt.utils.numpy import clump_nonzero_inclusive
    >>> clump_nonzero_inclusive([[0, 0, 1, 1, 0, 0]])
    [array([[2, 3]])]
    >>> clump_nonzero_inclusive([[1, 1, 0, 0, 1, 1]])
    [array([[0, 1],
           [4, 5]])]
    >>> clump_nonzero_inclusive([[1, 1, 1, 1, 1, 1]])
    [array([[0, 5]])]
    >>> clump_nonzero_inclusive([[0, 0, 0, 0, 0, 0]])
    [array([], shape=(0, 2), dtype=int64)]

    See also
    --------
    m4opt.utils.numpy.clump_nonzero
    """
    result = clump_nonzero(a)
    for intervals in result:
        intervals[:, 1] -= 1
    return result


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
