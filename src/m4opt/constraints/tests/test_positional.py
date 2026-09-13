import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, Angle, GeocentricTrueEcliptic, HADec, get_sun
from hypothesis import given, settings
from hypothesis import strategies as st

from ...dynamics import nominal_roll
from ...tests.hypothesis import (
    earth_locations,
    obstimes,
    skycoords,
)
from .._positional import (
    AltitudeConstraint,
    AzimuthConstraint,
    DeclinationConstraint,
    EclipticLatitudeConstraint,
    HelioeclipticLongitudeConstraint,
    HourAngleConstraint,
    NominalRollConstraint,
    RightAscensionConstraint,
)


def angle_deg(value):
    return value * u.deg


def interval_is_proper(min_max):
    min, max = min_max
    return min < max


def interval_is_distinct(min_max):
    min, max = min_max
    return min != max


def angle_bounds(min, max):
    return st.lists(
        st.floats(min, max, allow_nan=False, allow_subnormal=False).map(angle_deg),
        min_size=2,
        max_size=2,
    )


@settings(deadline=None)
@given(
    earth_locations,
    skycoords,
    obstimes,
    angle_bounds(0, 360).filter(interval_is_distinct),
    angle_bounds(-90, 90).filter(interval_is_proper),
)
def test_positional(observer_location, target_coord, obstime, lon_bounds, lat_bounds):
    lon_lo, lon_hi = lon_bounds
    lat_lo, lat_hi = lat_bounds
    args = observer_location, target_coord, obstime
    lon_lo = Angle(lon_lo).wrap_at(lon_hi)

    frame = target_coord.icrs
    lon = frame.ra.wrap_at(lon_hi)
    lat = frame.dec
    assert (lon_lo <= lon) & (lon <= lon_hi) == RightAscensionConstraint(*lon_bounds)(
        *args
    )
    assert (lat_lo <= lat) & (lat <= lat_hi) == DeclinationConstraint(*lat_bounds)(
        *args
    )

    frame = target_coord.transform_to(
        AltAz(obstime=obstime, location=observer_location)
    )
    lon = frame.az.wrap_at(lon_hi)
    lat = frame.alt
    assert (lon_lo <= lon) & (lon <= lon_hi) == AzimuthConstraint(*lon_bounds)(*args)
    assert (lat_lo <= lat) & (lat <= lat_hi) == AltitudeConstraint(*lat_bounds)(*args)

    frame = target_coord.transform_to(
        HADec(obstime=obstime, location=observer_location)
    )
    lon = frame.ha.wrap_at(lon_hi)
    assert (lon_lo <= lon) & (lon <= lon_hi) == HourAngleConstraint(*lon_bounds)(*args)

    frame = target_coord.transform_to(GeocentricTrueEcliptic(obstime=obstime))
    lat = frame.lat
    assert (lat_lo <= lat) & (lat <= lat_hi) == EclipticLatitudeConstraint(*lat_bounds)(
        *args
    )


@settings(deadline=None)
@given(
    earth_locations,
    skycoords,
    obstimes,
    angle_bounds(0, 180).filter(interval_is_proper),
)
def test_helioecliptic_longitude(observer_location, target_coord, obstime, lon_bounds):
    lon_lo, lon_hi = lon_bounds
    args = observer_location, target_coord, obstime

    frame = target_coord.transform_to(GeocentricTrueEcliptic(obstime=obstime))
    lon_target = frame.lon
    lon0 = get_sun(obstime).transform_to(frame).lon
    lon = np.abs((lon_target - lon0).wrap_at(180 * u.deg))
    assert (lon_lo <= lon) & (lon <= lon_hi) == HelioeclipticLongitudeConstraint(
        *lon_bounds
    )(*args)


@settings(deadline=None)
@given(
    earth_locations,
    skycoords,
    obstimes,
    angle_bounds(-180, 180).filter(interval_is_distinct),
    st.integers(1, 20),
)
def test_nominal_roll(observer_location, target_coord, obstime, bounds, symmetry):
    min, max = bounds

    constraint = NominalRollConstraint(min, max, symmetry)
    result = constraint(observer_location, target_coord, obstime)

    roll = (
        nominal_roll(observer_location, target_coord, obstime)[..., np.newaxis]
        + np.linspace(0, 360, symmetry, endpoint=False) * u.deg
    )
    while (roll > 180 * u.deg).any():
        roll[roll > 180 * u.deg] -= 360 * u.deg
    turn = 360 * u.deg
    roll_wrapped = np.where(roll > max, roll - turn, roll)
    constraint = NominalRollConstraint(min, max, symmetry)
    expected = np.where(
        min < max,
        (min <= roll) & (roll <= max),
        (min - turn <= roll_wrapped) & (roll_wrapped <= max),
    ).any(axis=-1)

    np.testing.assert_array_equal(result, expected)
