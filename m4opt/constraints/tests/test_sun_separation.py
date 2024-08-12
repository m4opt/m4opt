from warnings import catch_warnings, simplefilter

from astroplan import Observer
from astroplan import SunSeparationConstraint as AstroplanSunSeparationConstraint
from astropy import units as u
from astropy.coordinates import NonRotationTransformationWarning
from hypothesis import given, settings
from hypothesis import strategies as st

from ..sun_separation import SunSeparationConstraint
from .conftest import earth_locations, obstimes, skycoords


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes, st.floats(0, 180))
def test_astroplan(observer_location, target_coord, obstime, min_sep_deg):
    """Test that the sun separation constraint matches Astroplan's."""
    min_sep = min_sep_deg * u.deg
    constraint = SunSeparationConstraint(min_sep)
    astroplan_constraint = AstroplanSunSeparationConstraint(min_sep)
    result = constraint(observer_location, target_coord, obstime)
    with catch_warnings():
        # FIXME: In Python >= 3.11, we can pass the ingnore and category
        # keyword arguments to catch_warnings and then remove this line.
        simplefilter(action="ignore", category=NonRotationTransformationWarning)
        expected = astroplan_constraint(
            Observer(observer_location), target_coord, obstime
        )
    assert result == expected
