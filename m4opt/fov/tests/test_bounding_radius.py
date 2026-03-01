import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion, PolygonSkyRegion, RectangleSkyRegion, Regions

from m4opt.fov import bounding_radius


def test_circle():
    """CircleSkyRegion at the origin returns its radius."""
    region = CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg)
    result = bounding_radius(region)
    assert u.isclose(result, 3 * u.deg)


def test_circle_offset():
    """CircleSkyRegion offset from origin returns separation + radius."""
    region = CircleSkyRegion(SkyCoord(1 * u.deg, 0 * u.deg), 3 * u.deg)
    result = bounding_radius(region)
    assert u.isclose(result, 4 * u.deg)


def test_rectangle():
    """RectangleSkyRegion returns the half-diagonal."""
    region = RectangleSkyRegion(
        SkyCoord(0 * u.deg, 0 * u.deg), 6 * u.deg, 8 * u.deg
    )
    result = bounding_radius(region)
    assert u.isclose(result, 5 * u.deg, atol=0.1 * u.deg)


def test_compound():
    """Compound Regions returns max over sub-regions."""
    regions = Regions(
        [
            CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 3 * u.deg),
            CircleSkyRegion(SkyCoord(0 * u.deg, 0 * u.deg), 5 * u.deg),
        ]
    )
    result = bounding_radius(regions)
    assert u.isclose(result, 5 * u.deg)


def test_rubin_fov():
    """Rubin FOV returns approximately 2 deg."""
    from m4opt.missions import rubin

    result = bounding_radius(rubin.fov)
    assert 1.5 * u.deg < result < 2.5 * u.deg
