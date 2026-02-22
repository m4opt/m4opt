"""Tests for the earthshine background model."""

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from .... import observing
from .. import EarthshineBackground


def test_earthshine_positive():
    """EarthshineBackground returns positive flux across its wavelength range."""
    spec = EarthshineBackground()
    for wave in [1500, 2600, 5000, 8000, 10000]:
        assert spec(wave * u.AA).value > 0


def test_earthshine_regression():
    """Flux at specific wavelengths matches frozen values from the ECSV data."""
    spec = EarthshineBackground()
    np.testing.assert_almost_equal(
        spec(2600 * u.AA).value, 7.355851128493407e-11, decimal=16
    )
    np.testing.assert_almost_equal(
        spec(5000 * u.AA).value, 6.619863286318664e-07, decimal=12
    )


def test_earthshine_uv_fainter_than_visible():
    """Earthshine is reflected sunlight, so UV should be fainter than visible."""
    spec = EarthshineBackground()
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
