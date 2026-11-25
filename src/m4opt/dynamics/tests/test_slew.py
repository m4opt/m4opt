import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, CartesianRepresentation, SkyCoord
from astropy.tests.helper import assert_quantity_allclose
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from scipy.spatial.transform import Rotation

from .._slew import AngularMotionProfile, EigenAxisSlew


def _sky_offset_frame_separation(
    center1: SkyCoord,
    center2: SkyCoord,
    roll1: u.Quantity[u.physical.angle],
    roll2: u.Quantity[u.physical.angle],
) -> Angle:
    """Independent oracle for EigenAxisSlew.separation using SkyOffsetFrame."""
    frame1 = center1.skyoffset_frame(roll1)
    frame2 = center2.skyoffset_frame(roll2)
    basis = CartesianRepresentation(np.eye(3))
    r = SkyCoord(basis, frame=frame2).transform_to(frame1).cartesian.xyz.value
    theta = Rotation.from_matrix(r).magnitude()
    return Angle(theta * u.rad).to(u.deg)


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
    lon1=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    lat1=st.floats(min_value=-89, max_value=89, allow_nan=False),
    lon2=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    lat2=st.floats(min_value=-89, max_value=89, allow_nan=False),
    roll1=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    roll2=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
)
def test_eigen_axis_slew_separation_matches_sky_offset_frame(
    lon1, lat1, lon2, lat2, roll1, roll2
):
    """Cross-check against an independent oracle built from SkyOffsetFrame."""
    center1 = SkyCoord(lon1 * u.deg, lat1 * u.deg)
    center2 = SkyCoord(lon2 * u.deg, lat2 * u.deg)
    roll1 = roll1 * u.deg
    roll2 = roll2 * u.deg
    separation = EigenAxisSlew.separation(center1, center2, roll1, roll2)
    assert_quantity_allclose(
        separation,
        _sky_offset_frame_separation(center1, center2, roll1, roll2),
        atol=1e-6 * u.deg,
    )


@example(lon1=0, lat1=-51, roll1=60, roll2=-60)  # issue reproducer (x86_64)
@example(lon1=175, lat1=27, roll1=291, roll2=-291)  # issue reproducer (arm64)
@given(
    lon1=st.floats(min_value=0, max_value=360, allow_nan=False),
    lat1=st.floats(min_value=-89, max_value=89, allow_nan=False),
    roll1=st.floats(min_value=-360, max_value=360, allow_nan=False),
    roll2=st.floats(min_value=-360, max_value=360, allow_nan=False),
)
def test_eigen_axis_slew_separation_180_deg(lon1, lat1, roll1, roll2):
    """A separation of exactly 180 deg should not raise or produce NaN.

    See https://github.com/m4opt/m4opt/issues/648.
    """
    center1 = SkyCoord(lon1 * u.deg, lat1 * u.deg)
    center2 = SkyCoord((lon1 + 180) * u.deg, -lat1 * u.deg)
    separation = EigenAxisSlew.separation(
        center1, center2, roll1 * u.deg, roll2 * u.deg
    )
    assert np.isfinite(separation)
    assert_quantity_allclose(separation, 180 * u.deg)


@given(
    lat1=st.floats(min_value=-89, max_value=89, allow_nan=False),
    roll=st.floats(min_value=-360, max_value=360, allow_nan=False),
    delta=st.floats(min_value=1e-15, max_value=1e-2, allow_nan=False),
)
def test_eigen_axis_slew_separation_small_angle(lat1, roll, delta):
    """A small separation should be resolved with full double precision."""
    center1 = SkyCoord(0 * u.deg, lat1 * u.deg)
    center2 = SkyCoord(delta * u.deg, lat1 * u.deg)
    separation = EigenAxisSlew.separation(center1, center2, roll * u.deg, roll * u.deg)
    assert_quantity_allclose(separation, delta * u.deg, atol=0 * u.deg, rtol=1e-9)


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
