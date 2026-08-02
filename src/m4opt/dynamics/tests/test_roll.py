import pytest
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    UnitSphericalRepresentation,
    get_body,
)
from hypothesis import given, settings

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._roll import nominal_roll


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes)
def test_nominal_roll(observer_location, target_coord, obstime):
    roll = nominal_roll(observer_location, target_coord, obstime)
    sun = get_body("sun", obstime, observer_location)
    spacecraft_frame = target_coord.transform_to(sun.frame).skyoffset_frame(roll)

    v_y = CartesianRepresentation(0, 1, 0)
    v_z = CartesianRepresentation(0, 0, 1)
    v_sun = sun.transform_to(spacecraft_frame).represent_as(UnitSphericalRepresentation)
    assert v_sun.dot(v_y).to_value(u.dimensionless_unscaled) == pytest.approx(0), (
        "The direction to the sun must be perpendicular to the constructed +Y axis"
    )
    assert v_z.dot(v_sun).to_value(u.dimensionless_unscaled) <= 0, (
        "The sun must form an obtuse angle to the +Z axis"
    )
