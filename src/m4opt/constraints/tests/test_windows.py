"""Checks for sampled visibility windows and exposure containment."""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from numpy.testing import assert_allclose

from .. import DeclinationConstraint, visibility_windows

EPOCH = Time("2025-01-01", scale="tai")
LOCATION = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
TARGETS = SkyCoord([0, 1] * u.deg, [0, 1] * u.deg)


def sampled(values, times, **kwargs):
    return visibility_windows(
        lambda location, targets, obstime: values,
        LOCATION,
        TARGETS,
        EPOCH + times * u.s,
        **kwargs,
    )


def seconds(windows):
    return [(window - EPOCH).to_value(u.s) for window in windows]


def test_runs_and_nonuniform_grid():
    result = sampled(
        [[True, True, False, True, True], [False, True, False, False, False]],
        np.array([0, 2, 5, 9, 15]),
    )
    a, b = seconds(result)
    assert_allclose(a, [[0, 2], [9, 15]], atol=1e-9)
    assert_allclose(b, [[2, 2]], atol=1e-9)
    assert result[0].scale == "tai"


def test_duration_and_margin():
    a, b = seconds(
        sampled(True, np.array([0, 5, 10]), time_margin=2 * u.s, min_duration=6 * u.s)
    )
    assert_allclose(a, [[2, 8]], atol=1e-9)
    assert_allclose(b, [[2, 8]], atol=1e-9)
    assert all(
        w.shape == (0, 2)
        for w in sampled(True, np.array([0, 5, 10]), time_margin=6 * u.s)
    )


def test_never_and_always_visible():
    a, b = sampled([[False, False], [True, True]], np.array([0, 10]))
    assert a.shape == (0, 2)
    assert_allclose((b - EPOCH).to_value(u.s), [[0, 10]], atol=1e-9)


def test_scalar_target_and_time_independent_constraint():
    constraint = DeclinationConstraint(-0.5 * u.deg, 0.5 * u.deg)
    times = EPOCH + np.array([0, 10]) * u.s
    result = visibility_windows(constraint, LOCATION, TARGETS, times)
    assert [w.shape for w in result] == [(1, 2), (0, 2)]
    result = visibility_windows(constraint, LOCATION, TARGETS[0], times)
    assert len(result) == 1
    assert_allclose(seconds(result)[0], [[0, 10]], atol=1e-9)


def test_empty_and_single_sample():
    assert all(w.shape == (0, 2) for w in sampled(True, np.array([])))
    assert all(w.shape == (1, 2) for w in sampled(True, np.array([0])))
    assert all(
        w.shape == (0, 2) for w in sampled(True, np.array([0]), min_duration=1 * u.s)
    )
    assert (
        visibility_windows(
            lambda *args: True, LOCATION, TARGETS[:0], EPOCH + np.array([0, 1]) * u.s
        )
        == []
    )


@pytest.mark.parametrize("times", [[0, 0], [1, 0]])
def test_invalid_grid(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        sampled(True, np.array(times))


@pytest.mark.parametrize("name", ["min_duration", "time_margin"])
@pytest.mark.parametrize("value", [-1 * u.s, np.inf * u.s, [1, 2] * u.s])
def test_invalid_duration(name, value):
    with pytest.raises(ValueError, match="scalar duration"):
        sampled(True, np.array([0, 1]), **{name: value})


def test_exposure_cannot_bridge_interruption():
    result = sampled(
        [True, True, False, True, True], np.arange(5), min_duration=2 * u.s
    )
    assert all(w.shape == (0, 2) for w in result)


def test_agrees_with_scheduler_extraction():
    from ...utils.numpy import clump_nonzero_inclusive

    rng = np.random.default_rng(1234)
    times = np.arange(40) * 30
    mask = rng.random((2, len(times))) > 0.25
    actual = seconds(sampled(mask, times, min_duration=45 * u.s))
    for row, indices in zip(actual, clump_nonzero_inclusive(mask)):
        expected = times[indices]
        expected = expected[np.diff(expected, axis=1).ravel() >= 45]
        assert_allclose(row, expected, atol=1e-9)


def test_masked_constraint():
    with pytest.raises(ValueError, match="unmasked visibility"):
        sampled(np.ma.array([True, True], mask=[False, True]), np.array([0, 1]))


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="obstime must be one-dimensional"):
        visibility_windows(lambda *args: True, LOCATION, TARGETS, EPOCH)
    with pytest.raises(ValueError, match="target_coord must be"):
        visibility_windows(
            lambda *args: True,
            LOCATION,
            TARGETS[:, None],
            EPOCH + np.array([0, 1]) * u.s,
        )


def test_duration_units():
    with pytest.raises(u.UnitConversionError):
        sampled(True, np.array([0, 1]), min_duration=1 * u.deg)
