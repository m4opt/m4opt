"""Tests for the earthshine background model."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.time import Time

from .... import observing
from .. import EarthshineBackground, EarthshineBackgroundScaleFactor

# On 2025-01-01, the Sun is at RA~281 deg, Dec~-23 deg.
# Use RA near the Sun for scale factor tests so the illumination
# factor is close to 1.0 and doesn't confound the limb angle tests.
_TEST_OBSTIME = Time("2025-01-01T00:00:00Z")
_SUN_RA_DEG = 281.0


def test_earthshine_high_positive():
    """EarthshineBackground.high() returns positive flux across its wavelength range."""
    spec = EarthshineBackground.high()
    for wave in [1500, 2600, 5000, 8000, 10000]:
        assert spec(wave * u.AA).value > 0


def test_earthshine_high_regression():
    """Flux at specific wavelengths matches frozen values from the ECSV data."""
    spec = EarthshineBackground.high()
    np.testing.assert_almost_equal(
        spec(2600 * u.AA).value, 7.355851128493407e-11, decimal=16
    )
    np.testing.assert_almost_equal(
        spec(5000 * u.AA).value, 6.619863286318664e-07, decimal=12
    )


def test_earthshine_high_uv_fainter_than_visible():
    """Earthshine is reflected sunlight, so UV should be fainter than visible."""
    spec = EarthshineBackground.high()
    uv = spec(2600 * u.AA).value
    vis = spec(5000 * u.AA).value
    assert uv < vis


def test_earthshine_in_context():
    """Works within observing() context."""
    loc = EarthLocation.from_geodetic(
        lon=15 * u.deg, lat=0 * u.deg, height=35786 * u.km
    )
    coord = SkyCoord(0 * u.deg, 0 * u.deg)
    obstime = Time("2025-05-18T02:48:00Z")

    bg = EarthshineBackground()
    with observing(observer_location=loc, target_coord=coord, obstime=obstime):
        val = bg(2600 * u.AA)
    # FIXME: add a more constraining test (e.g., regression value check)
    assert val.value > 0


def test_earthshine_requires_context():
    """EarthshineBackground() raises ValueError without observing() context."""
    bg = EarthshineBackground()
    with pytest.raises(ValueError, match="Unknown target"):
        bg(5000 * u.AA)


def test_earthshine_scale_factor_calibration_points():
    """Scale factor reproduces calibration points at 24, 38, and 50 degrees."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth from center (altitude = R_earth above surface).
    # limb_alt = arccos(R_earth / (2*R_earth)) = 60 deg.
    # limb_angle = alt + 60 deg, so alt = limb_angle - 60 deg.
    # At the North Pole, alt = dec.
    # Use RA near the Sun so illumination factor ~ 1.
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    for limb_angle_deg, expected_scale in [(24, 2.0), (38, 1.0), (50, 0.5)]:
        alt_deg = limb_angle_deg - 60
        coord = SkyCoord(_SUN_RA_DEG * u.deg, alt_deg * u.deg)
        scale = sf.at(loc, coord, _TEST_OBSTIME)
        np.testing.assert_allclose(scale, expected_scale, rtol=0.15)


def test_earthshine_scale_factor_below_limb():
    """Scale factor is zero for targets below the Earth's limb."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    # Target at alt = -90 deg => limb_angle = -90 + 60 = -30 deg (below limb).
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    coord = SkyCoord(_SUN_RA_DEG * u.deg, -90 * u.deg)

    scale = sf.at(loc, coord, _TEST_OBSTIME)
    assert scale == 0.0


def test_earthshine_scale_factor_clamps_near_limb():
    """Scale factor clamps at 2.0 for targets between 0 and 24 deg from limb."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    # Target just above limb (limb_angle ~ 5 deg, below 24 deg)
    # alt = 5 - 60 = -55 deg
    coord_near = SkyCoord(_SUN_RA_DEG * u.deg, -55 * u.deg)
    scale_near = sf.at(loc, coord_near, _TEST_OBSTIME)
    np.testing.assert_allclose(scale_near, 2.0, rtol=0.15)


def test_earthshine_scale_factor_extrapolates_far_from_limb():
    """Scale factor extrapolates to small values far from the Earth limb."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    # Target at limb_angle = 80 deg (well above 50 deg calibration point).
    # alt = 80 - 60 = 20 deg
    coord_far = SkyCoord(_SUN_RA_DEG * u.deg, 20 * u.deg)
    scale_far = sf.at(loc, coord_far, _TEST_OBSTIME)
    # Should be much less than 0.5 (the value at 50 deg) due to extrapolation
    assert scale_far < 0.1


def test_earthshine_scale_factor_monotonic():
    """Scale factor is monotonically decreasing with increasing limb angle."""
    sf = EarthshineBackgroundScaleFactor()

    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    # Test limb angles from 24 deg to 120 deg.
    # Use RA near the Sun so illumination is roughly constant.
    # Start from 24 deg (below that, the limb component is clamped at 2.0
    # and the illumination variation can break strict monotonicity).
    limb_angles = [24, 30, 38, 50, 60, 80, 100, 120]
    scales = []
    for la in limb_angles:
        alt_deg = la - 60
        coord = SkyCoord(_SUN_RA_DEG * u.deg, alt_deg * u.deg)
        scales.append(sf.at(loc, coord, _TEST_OBSTIME))

    for i in range(1, len(scales)):
        assert scales[i] <= scales[i - 1]


def test_earthshine_illumination_sunlit_vs_dark():
    """Earthshine is stronger when looking toward the sunlit side of Earth."""
    sf = EarthshineBackgroundScaleFactor()

    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)

    # Use the same limb angle (38 deg => alt = -22 deg) but different RAs:
    # one near the Sun (sunlit limb) and one opposite (dark limb).
    sun = get_sun(_TEST_OBSTIME)
    alt_deg = -22  # limb angle = 38 deg

    coord_sunlit = SkyCoord(sun.ra, alt_deg * u.deg)
    coord_dark = SkyCoord(sun.ra + 180 * u.deg, alt_deg * u.deg)

    scale_sunlit = sf.at(loc, coord_sunlit, _TEST_OBSTIME)
    scale_dark = sf.at(loc, coord_dark, _TEST_OBSTIME)

    assert scale_sunlit > 0
    assert scale_dark >= 0
    assert scale_sunlit > 2 * scale_dark
