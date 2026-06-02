from dataclasses import dataclass

import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import composite, floats
from numpy.polynomial import polyutils as pu
from numpy.polynomial.polynomial import polyval

from ..interp import athena_interp


# FIXME: https://github.com/numpy/numpy/issues/30857
def polyvalnd(x, c):
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
    regular: bool = False,
):
    shape = draw(array_shapes(min_dims=min_dims, max_dims=max_dims, min_side=order + 1))
    ndim = len(shape)

    if regular:
        points = [
            draw(floats(-10000, 10000, allow_nan=False, allow_infinity=False))
            + np.arange(n)
            * draw(floats(1e-3, 10000, allow_nan=False, allow_infinity=False))
            for n in shape
        ]
    else:
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
            shape=[order] * ndim,
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


@given(polynomial_sample_data(order=3, regular=True))
def test_athena_interp(data):
    """Test the interpolation scheme using data from a multivariate polynomial
    of degree that matches the order of the interpolation scheme."""
    lo = np.asarray([pt.min() for pt in data.points])
    hi = np.asarray([pt.max() for pt in data.points])
    delta = np.asarray([pt[1] - pt[0] for pt in data.points])
    ndim = len(data.points)
    values = polyvalnd(np.meshgrid(*data.points, indexing="ij"), data.poly)
    result = athena_interp(data.points, values, data.xi)

    assert result.shape == (*(data.xi.shape[:-1] or (1,)), *values.shape[ndim:]), (
        "The shape of the output must match what would have been returned by scipy.interpolate.interpn. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html"
    )

    out_of_bounds = ((data.xi < lo) & (data.xi > hi)).any(axis=-1)
    assert np.isnan(result[out_of_bounds]).all(), (
        "Interpolant must retern NaN for all out-of-bounds input points"
    )

    xi_transpose = np.moveaxis(data.xi, -1, 0)
    exact_polynomial = np.atleast_1d(polyvalnd(xi_transpose, data.poly))
    in_bounds_with_padding = ((data.xi >= lo + delta) & (data.xi <= hi - delta)).all(
        axis=-1
    )
    np.testing.assert_allclose(
        result[in_bounds_with_padding],
        exact_polynomial[in_bounds_with_padding],
        atol=1e-6,
        err_msg="Interpolant must exactly match the function everywhere except within 1 sample point of any boundary",
    )

    # FIXME: Add test for points near boundary
