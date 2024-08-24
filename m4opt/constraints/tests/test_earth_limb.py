import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import AltAz
from hypothesis import given, settings

from ...tests.hypothesis import (
    earth_locations_at_geocentric_radius,
    obstimes,
    skycoords,
)
from .._earth_limb import _get_angle_from_earth_limb


@settings(deadline=None)
@given(earth_locations_at_geocentric_radius(0.9 * R_earth), skycoords, obstimes)
def test_observer_beneath_earth_surface(observer_location, target_coord, obstime):
    """Test angle from earth limb for observer beneath surface (must be NaN)"""
    result = _get_angle_from_earth_limb(observer_location, target_coord, obstime)
    assert np.isnan(result)


@settings(deadline=None)
@given(earth_locations_at_geocentric_radius(R_earth), skycoords, obstimes)
def test_observer_on_earth(observer_location, target_coord, obstime):
    """Test angle from earth limb for observer on surface (must equal altitude angle)"""
    expected = target_coord.transform_to(
        AltAz(location=observer_location, obstime=obstime)
    ).alt.to_value(u.deg)
    result = _get_angle_from_earth_limb(
        observer_location, target_coord, obstime
    ).to_value(u.deg)
    assert np.isnan(result) or result == pytest.approx(expected, abs=1e-6)


@settings(deadline=None)
@given(earth_locations_at_geocentric_radius(2 * R_earth), skycoords, obstimes)
def test_observer_1_rearth_above_surface(observer_location, target_coord, obstime):
    """Test angle from earth limb for observer 1 R_earth above surface (must equal 60° plus altitude angle)"""
    expected = (
        target_coord.transform_to(
            AltAz(location=observer_location, obstime=obstime)
        ).alt.to_value(u.deg)
        + 60
    )
    result = _get_angle_from_earth_limb(
        observer_location, target_coord, obstime
    ).to_value(u.deg)
    assert result == pytest.approx(expected, abs=1e-6)
