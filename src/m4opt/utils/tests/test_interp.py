from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import composite, floats
from numpy.polynomial.polynomial import polyvalnd

from ..interp import athena_interp


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

    out_of_bounds = ((data.xi < lo) | (data.xi > hi)).any(axis=-1)
    assert np.isnan(result[out_of_bounds]).all(), (
        "Interpolant must retern NaN for all out-of-bounds input points"
    )

    xi_transpose = np.moveaxis(data.xi, -1, 0)
    exact_polynomial = np.atleast_1d(polyvalnd(xi_transpose, data.poly))
    in_bounds_with_padding = ((data.xi >= lo + delta) & (data.xi <= hi - delta)).all(
        axis=-1
    )
    # A value that cancels to zero carries no relative scale of its own, so
    # rounding is allowed at the magnitude of the samples.
    finite = np.abs(values[np.isfinite(values)])
    np.testing.assert_allclose(
        result[in_bounds_with_padding],
        exact_polynomial[in_bounds_with_padding],
        rtol=1e-5,
        atol=1e-12 * (finite.max() if finite.size else 0.0),
        err_msg="Interpolant must exactly match the function everywhere except within 1 sample point of any boundary",
    )

    # FIXME: Add test for points near boundary


@pytest.mark.parametrize("scale", [1e-4, 1e-6, 1e-8, 1e-179])
def test_athena_interp_is_scale_invariant(scale):
    """Scaling the values scales the result, because interpolation is linear."""
    points = [np.array([0.0, 1.0, 2.0, 3.0])]
    values = np.array([1.0, 3.0, 7.0, 13.0])
    xi = np.array([[1.5]])
    expected = athena_interp(points, values, xi) * scale
    np.testing.assert_allclose(
        athena_interp(points, values * scale, xi), expected, rtol=1e-12
    )


def test_athena_interp_reproduces_the_grid_values():
    """At a grid point the interpolant returns that point's own value."""
    points = [np.arange(-2.0, 7.0)]
    values = 1 + points[0] + points[0] ** 2
    result = athena_interp(points, values, points[0][:, np.newaxis])
    np.testing.assert_allclose(result, values, rtol=1e-12)


def test_athena_interp_is_nan_out_of_bounds():
    """A sample beyond either end of the grid has no interpolated value."""
    points = [np.array([0.0, 1.0, 2.0, 3.0])]
    values = np.array([1.0, 3.0, 7.0, 13.0])
    result = athena_interp(points, values, np.array([[-1.0], [1.5], [4.0]]))
    assert np.isnan(result[[0, 2]]).all()
    assert np.isfinite(result[1])
