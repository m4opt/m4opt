from warnings import catch_warnings

import pytest
from astroplan import MoonSeparationConstraint as AstroplanMoonSeparationConstraint
from astroplan import Observer
from astroplan import SunSeparationConstraint as AstroplanSunSeparationConstraint
from astropy import units as u
from astropy.coordinates import NonRotationTransformationWarning, get_body
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._body_separation import (
    AntiSolarSeparationConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)


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


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes, st.floats(0, 180))
def test_anti_solar_separation(observer_location, target_coord, obstime, min_sep_deg):
    """Test that AntiSolarSeparationConstraint agrees with 180° minus the
    solar elongation."""
    min_sep = min_sep_deg * u.deg
    with catch_warnings(action="ignore", category=NonRotationTransformationWarning):
        sun_separation = get_body(
            "sun", time=obstime, location=observer_location
        ).separation(target_coord, origin_mismatch="ignore")
    anti_solar_separation = 180 * u.deg - sun_separation
    # Avoid the boundary, where floating point error could flip the result.
    assume(abs(anti_solar_separation - min_sep) > 1e-6 * u.deg)
    constraint = AntiSolarSeparationConstraint(min_sep)
    result = constraint(observer_location, target_coord, obstime)
    expected = anti_solar_separation >= min_sep
    assert result == expected
