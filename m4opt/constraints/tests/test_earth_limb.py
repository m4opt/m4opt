import numpy as np
import pytest
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, SphericalRepresentation
from astropy.time import Time
from hypothesis import given, settings
from hypothesis import strategies as st

from ..earth_limb import _get_angle_from_earth_limb

radecs = st.tuples(st.floats(0, 2 * np.pi), st.floats(-np.pi / 2, np.pi / 2))
skycoords = radecs.map(lambda radec: SkyCoord(*radec, unit=u.rad))
obstimes = st.floats(60310, 60676).map(lambda mjd: Time(mjd, format="mjd"))


def earth_locations(geocentric_radius):
    return radecs.map(
        lambda lonlat: EarthLocation.from_geocentric(
            *SphericalRepresentation(
                lonlat[0] * u.rad, lonlat[1] * u.rad, geocentric_radius
            )
            .to_cartesian()
            .xyz
        )
    )


@settings(deadline=None)
@given(earth_locations(0.9 * R_earth), skycoords, obstimes)
def test_observer_beneath_earth_surface(observer_location, target_coord, obstime):
    """Test angle from earth limb for observer beneath surface (must be NaN)"""
    result = _get_angle_from_earth_limb(observer_location, target_coord, obstime)
    assert np.isnan(result)


@settings(deadline=None)
@given(earth_locations(R_earth), skycoords, obstimes)
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
@given(earth_locations(2 * R_earth), skycoords, obstimes)
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
