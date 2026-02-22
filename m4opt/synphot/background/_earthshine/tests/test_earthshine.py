"""Tests for the earthshine background model."""

import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .... import observing
from .. import EarthshineBackground, EarthshineBackgroundScaleFactor


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
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    obstime = Time("2025-01-01T00:00:00Z")

    for limb_angle_deg, expected_scale in [(24, 2.0), (38, 1.0), (50, 0.5)]:
        # Target at altitude = limb_angle - 60 deg (in AltAz frame, the
        # target needs altitude such that alt + limb_alt = limb_angle).
        # At the North Pole with geocentric z-axis observer, altitude
        # maps directly to declination offset.
        alt_deg = limb_angle_deg - 60
        coord = SkyCoord(0 * u.deg, (90 + alt_deg) * u.deg)
        scale = sf.at(loc, coord, obstime)
        np.testing.assert_allclose(scale, expected_scale, rtol=0.15)


def test_earthshine_scale_factor_below_limb():
    """Scale factor is zero for targets below the Earth's limb."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    # Target at alt = -70 deg => limb_angle = -70 + 60 = -10 deg (below limb).
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    obstime = Time("2025-01-01T00:00:00Z")
    # Point toward nadir (declination = -90 from zenith)
    coord = SkyCoord(0 * u.deg, -90 * u.deg)

    scale = sf.at(loc, coord, obstime)
    assert scale == 0.0


def test_earthshine_scale_factor_clamping():
    """Scale factor clamps at boundaries (2.0 for <24 deg, 0.5 for >50 deg)."""
    sf = EarthshineBackgroundScaleFactor()

    # Observer at 2*R_earth, limb_alt = 60 deg.
    loc = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    obstime = Time("2025-01-01T00:00:00Z")

    # Target at high altitude (well above 50 deg from limb)
    # alt = 80 deg => limb_angle = 80 + 60 = 140 deg (way above 50)
    # But with our setup, alt = limb_angle - 60, so for 80 deg limb angle:
    # alt = 20 deg
    coord_far = SkyCoord(0 * u.deg, (90 + 20) * u.deg)
    scale_far = sf.at(loc, coord_far, obstime)
    np.testing.assert_allclose(scale_far, 0.5, rtol=0.15)

    # Target just above limb (limb_angle ~ 5 deg, below 24 deg)
    # alt = 5 - 60 = -55 deg
    coord_near = SkyCoord(0 * u.deg, (90 - 55) * u.deg)
    scale_near = sf.at(loc, coord_near, obstime)
    np.testing.assert_allclose(scale_near, 2.0, rtol=0.15)
