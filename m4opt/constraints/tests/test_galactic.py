import numpy as np
from astropy import units as u
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._galactic import GalacticLatitudeConstraint


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes, st.floats(0, 90))
def test_galactic_latitude_constraint(
    observer_location, target_coord, obstime, min_deg
):
    expected = np.abs(target_coord.galactic.b.deg) >= min_deg
    result = GalacticLatitudeConstraint(min_deg * u.deg)(
        observer_location, target_coord, obstime
    )
    assert result == expected
