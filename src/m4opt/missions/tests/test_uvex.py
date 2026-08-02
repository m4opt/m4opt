import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import (
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
    get_body,
)
from hypothesis import given, settings

from ...tests.hypothesis import obstimes
from .._uvex import uvex as mission
from .._uvex import uvex_downlink_orientation


@settings(deadline=None)
@given(obstimes)
def test_uvex_downlink_orientation(obstime):
    observer_location = mission.observer_location(obstime)
    earth = get_body("earth", obstime, observer_location)
    sun = get_body("sun", obstime, observer_location)
    target_coord, roll = uvex_downlink_orientation(obstime)
    spacecraft_frame = target_coord.transform_to(sun.frame).skyoffset_frame(roll)
    antenna = SkyCoord(
        CartesianRepresentation(-np.sqrt(2) / 2, 0, np.sqrt(2) / 2),
        frame=spacecraft_frame,
    )

    v_y = CartesianRepresentation(0, 1, 0)
    v_sun = sun.transform_to(spacecraft_frame).represent_as(UnitSphericalRepresentation)
    assert v_sun.dot(v_y).to_value(u.dimensionless_unscaled) == pytest.approx(0), (
        "The direction to the sun must be perpendicular to the constructed +Y axis"
    )

    assert target_coord.separation(sun) >= 45 * u.deg, (
        "The direction to the sun must be at least 45° from the +X axis"
    )

    assert antenna.separation(earth).to_value(u.deg) == pytest.approx(0), (
        "The antenna must point toward the Earth"
    )

    assert target_coord.separation(earth).to_value(u.deg) == pytest.approx(135), (
        "The angle from the Earth to the target must equal 135°"
    )
