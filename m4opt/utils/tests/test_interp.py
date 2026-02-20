from dataclasses import dataclass

import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import composite, floats
from numpy.polynomial import polyutils as pu
from numpy.polynomial.polynomial import polyval

from ..interp import athena_interp


def polyndval(x, c):
    """Evaluate an arbitrary multivariate polynomial."""
    return pu._valnd(polyval, c, *x)


@dataclass
class PolySampleData:
    """Polynomial sample data for testing interpolation in N dimensions."""

    points: list[np.ndarray]
    """A list of N 1-D arrays denoting the grid points to sample the function.
    Within each array, the values must be monotonically increasing."""

    poly: np.ndarray
    """A square array of coefficients of a multivariate polynomial of degree at
    most (N - 1)."""

    xi: np.ndarray
    """Array of sample points at which to test the interpolation. May be any
    shape as long as the trailing dimension is N."""


@composite
def polynomial_sample_data(
    draw,
    order: int,
    min_dims: int = 1,
    max_dims: int | None = None,
    max_broadcast_dims: int | None = None,
):
    num_coefficients = order + 1
    shape = draw(
        array_shapes(min_dims=min_dims, max_dims=max_dims, min_side=num_coefficients)
    )
    ndim = len(shape)

    points = [
        draw(
            arrays(
                dtype=np.float64,
                shape=dim,
                elements=floats(
                    allow_nan=False,
                    allow_infinity=False,
                    min_value=-10000,
                    max_value=10000,
                ),
                unique=True,
            ).map(np.sort)
        )
        for dim in shape
    ]

    poly = draw(
        arrays(
            dtype=np.float64,
            shape=[num_coefficients] * ndim,
            elements=floats(
                allow_nan=False, allow_infinity=False, min_value=-100, max_value=100
            ),
        )
    )

    xi = draw(
        arrays(
            dtype=np.float64,
            shape=(*draw(array_shapes(min_dims=0, max_dims=max_broadcast_dims)), ndim),
        )
    )
    return PolySampleData(points, poly, xi)


#
# FIXME for Athena:
# Change `order=1` to `order=3` for cubic polynomial sample data.
#
# Currently this will test univariate interpolation with scalar inputs.
# When you for a greater challenge, do the following:
#
#   - Remove the keyword argument `max_dims` to try multivariate interpolation.
#   - Remove the keyword argument `max_broadcast_dims` to try tensor inputs.
#
@given(polynomial_sample_data(order=1, max_dims=1, max_broadcast_dims=0))
def test_athena_interp(data):
    """Test the interpolation scheme using data from a multivariate polynomial
    of degree that matches the order of the interpolation scheme."""
    lo = np.asarray([pt.min() for pt in data.points])
    hi = np.asarray([pt.max() for pt in data.points])
    xi_transpose = np.moveaxis(data.xi, -1, 0)
    in_bounds = ((data.xi >= lo) & (data.xi <= hi)).all(axis=-1)
    desired = np.where(in_bounds, polyndval(xi_transpose, data.poly), np.nan)

    values = polyndval(np.meshgrid(*data.points, indexing="ij"), data.poly)
    actual = athena_interp(data.points, values, data.xi)

    np.testing.assert_array_almost_equal(actual, desired)
