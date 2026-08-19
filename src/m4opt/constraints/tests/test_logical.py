import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import earth_locations, obstimes, skycoords
from .._positional import DeclinationConstraint


@settings(deadline=None)
@given(earth_locations, skycoords, obstimes, st.floats(-90, 90), st.floats(-90, 90))
def test_logical(observer_location, target_coord, obstime, min_deg, max_deg):
    """Test that the logical operators agree with their Numpy counterparts."""
    args = observer_location, target_coord, obstime
    lhs = DeclinationConstraint(min_deg * u.deg, max_deg * u.deg)
    rhs = DeclinationConstraint(-90 * u.deg, 0 * u.deg)
    lhs_value = lhs(*args)
    rhs_value = rhs(*args)

    assert (~lhs)(*args) == np.logical_not(lhs_value)
    assert (lhs & rhs)(*args) == np.logical_and(lhs_value, rhs_value)
    assert (lhs | rhs)(*args) == np.logical_or(lhs_value, rhs_value)


def test_logical_not_is_elementwise():
    """Test that a logical "not" is evaluated for each target.

    This is a regression test to ensure that the operand is actually
    evaluated, rather than being silently replaced by a scalar.
    """
    observer_location = None
    target_coord = SkyCoord([0, 0, 0] * u.deg, [-80, 0, 80] * u.deg)
    obstime = None
    constraint = DeclinationConstraint(-30 * u.deg, 30 * u.deg)

    np.testing.assert_array_equal(
        constraint(observer_location, target_coord, obstime),
        [False, True, False],
    )
    np.testing.assert_array_equal(
        (~constraint)(observer_location, target_coord, obstime),
        [True, False, True],
    )
