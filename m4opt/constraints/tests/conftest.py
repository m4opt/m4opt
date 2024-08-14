import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord, SphericalRepresentation
from astropy.time import Time
from hypothesis import strategies as st

__all__ = (
    "obstimes",
    "radecs",
    "skycoords",
    "earth_locations",
    "earth_locations_at_geocentric_radius",
)

obstimes = st.floats(60310, 60676).map(lambda mjd: Time(mjd, format="mjd"))
radecs = st.tuples(st.floats(0, 2 * np.pi), st.floats(-np.pi / 2, np.pi / 2))
skycoords = radecs.map(lambda radec: SkyCoord(*radec, unit=u.rad))


earth_locations = st.tuples(
    st.floats(0, 2 * np.pi), st.floats(-np.pi / 2, np.pi / 2), st.floats(0, 10)
).map(
    lambda lonlatdist: EarthLocation.from_geocentric(
        *SphericalRepresentation(
            lonlatdist[0] * u.rad, lonlatdist[1] * u.rad, lonlatdist[2] * u.au
        )
        .to_cartesian()
        .xyz
    )
)


def earth_locations_at_geocentric_radius(radius: u.Quantity[u.physical.length]):
    return radecs.map(
        lambda lonlat: EarthLocation.from_geocentric(
            *SphericalRepresentation(lonlat[0] * u.rad, lonlat[1] * u.rad, radius)
            .to_cartesian()
            .xyz
        )
    )
