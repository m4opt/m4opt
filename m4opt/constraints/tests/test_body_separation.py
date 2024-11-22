from warnings import catch_warnings

import pytest
from astroplan import MoonSeparationConstraint as AstroplanMoonSeparationConstraint
from astroplan import Observer
from astroplan import SunSeparationConstraint as AstroplanSunSeparationConstraint
from astropy import units as u
from astropy.coordinates import NonRotationTransformationWarning
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._body_separation import MoonSeparationConstraint, SunSeparationConstraint


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes, st.floats(0, 180))
@pytest.mark.parametrize(
    ["cls", "astroplan_cls"],
    [
        [MoonSeparationConstraint, AstroplanMoonSeparationConstraint],
        [SunSeparationConstraint, AstroplanSunSeparationConstraint],
    ],
)
def test_astroplan(
    cls, astroplan_cls, observer_location, target_coord, obstime, min_sep_deg
):
    """Test that the constraint matches Astroplan's."""
    min_sep = min_sep_deg * u.deg
    constraint = cls(min_sep)
    astroplan_constraint = astroplan_cls(min_sep)
    result = constraint(observer_location, target_coord, obstime)
    with catch_warnings(action="ignore", category=NonRotationTransformationWarning):
        expected = astroplan_constraint(
            Observer(observer_location), target_coord, obstime
        )
    assert result == expected
