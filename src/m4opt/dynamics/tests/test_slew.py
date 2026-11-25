import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from .._slew import AngularMotionProfile, EigenAxisSlew


@given(
    arrays(
        float,
        array_shapes(min_dims=0, min_side=0),
        elements=st.floats(min_value=0, max_value=1e3),
    ),
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=1e-3),
    st.floats(min_value=0, max_value=1e3),
)
def test_angular_motion_profile_no_jerk(
    distance, max_angular_velocity, max_angular_acceleration, settling_time
):
    max_angular_velocity *= u.rad / u.s
    max_angular_acceleration *= u.rad / u.s**2
    settling_time *= u.s

    profile = AngularMotionProfile(
        max_angular_velocity=max_angular_velocity,
        max_angular_acceleration=max_angular_acceleration,
        settling_time=settling_time,
    )

    distance_roundtrip = profile._distance(profile._time(distance * u.rad)).to_value(
        u.rad
    )
    np.testing.assert_almost_equal(distance_roundtrip, distance)


@given(
    arrays(
        float,
        array_shapes(min_dims=0, min_side=0),
        elements=st.floats(min_value=0, max_value=1e3),
    ),
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=1e-3, max_value=1e3),
    st.floats(min_value=0, max_value=1e3),
)
def test_angular_motion_profile_jerk(
    distance,
    max_angular_velocity,
    max_angular_acceleration,
    max_angular_jerk,
    settling_time,
):
    max_angular_velocity *= u.rad / u.s
    max_angular_acceleration *= u.rad / u.s**2
    max_angular_jerk *= u.rad / u.s**3
    settling_time *= u.s

    profile = AngularMotionProfile(
        max_angular_velocity=max_angular_velocity,
        max_angular_acceleration=max_angular_acceleration,
        max_angular_jerk=max_angular_jerk,
        settling_time=settling_time,
    )

    distance_roundtrip = profile._distance(profile._time(distance * u.rad)).to_value(
        u.rad
    )
    np.testing.assert_almost_equal(distance_roundtrip, distance)


def test_eigen_axis_slew_separation_180_deg():
    """Rounding error used to give NaN for a rotation of exactly 180 deg."""
    center1 = SkyCoord(0 * u.deg, -65 * u.deg)
    center2 = SkyCoord(180 * u.deg, 65 * u.deg)
    separation = EigenAxisSlew.separation(center1, center2, 0 * u.deg, -30 * u.deg)
    assert np.isfinite(separation)
    assert_quantity_allclose(separation, 180 * u.deg)
