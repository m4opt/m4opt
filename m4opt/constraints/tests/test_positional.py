from astropy import units as u
from astropy.coordinates import AltAz, Angle, HADec
from hypothesis import given, settings
from hypothesis import strategies as st

from ...tests.hypothesis import (
    earth_locations,
    obstimes,
    skycoords,
)
from .._positional import (
    AltitudeConstraint,
    AzimuthConstraint,
    DeclinationConstraint,
    HourAngleConstraint,
    RightAscensionConstraint,
)


def angle_deg(value):
    return value * u.deg


def interval_is_proper(min_max):
    min, max = min_max
    return min < max


def angle_bounds(min, max):
    return st.lists(
        st.floats(min, max, allow_nan=False, allow_subnormal=False).map(angle_deg),
        min_size=2,
        max_size=2,
    ).filter(interval_is_proper)


@settings(deadline=None)
@given(
    earth_locations, skycoords, obstimes, angle_bounds(0, 360), angle_bounds(-90, 90)
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
