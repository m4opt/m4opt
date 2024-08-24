import numpy as np
from astropy import units as u
from astropy.coordinates import TEME, SkyCoord
from astropy.utils.data import get_readable_fileobj
from satellite_tle import fetch_tle_from_celestrak
from sgp4.api import SGP4_ERRORS, Satrec

from ._core import Orbit


class TLE(Orbit):
    """An Earth satellite whose orbit is specified by its TLE.

    Parameters
    ----------
    tle : str, file
        The filename or file-like object containing the two-line element (TLE).

    Examples
    --------

    Here is an example TLE for BurstCube (NORAD ID 59562):

    >>> from importlib import resources
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> import numpy as np
    >>> from m4opt.orbit import TLE
    >>> line1 = '1 59562U 98067WM  24220.55604657  .00200610  00000+0  13802-2 0  9999'
    >>> line2 = '2 59562  51.6321  52.1851 0005233 205.4266 154.6476 15.73479335 17375'
    >>> orbit = TLE(line1, line2)

    Get the orbital period:

    >>> orbit.period
    <Quantity 91.51693117 min>

    Evaluate the position and velocity of the satellite at one specific time:

    >>> time = Time('2024-08-08 01:10:41')
    >>> orbit(time)
    <SkyCoord (ITRS: obstime=2024-08-08 01:10:41.000, location=(0.0, 0.0, 0.0) km): (x, y, z) in km
        (4172.83527806, -514.30119564, -5254.79810552)
     (v_x, v_y, v_z) in km / s
        (1.5531795, 7.20225602, 0.52399427)>

    Or evaluate at an array of times:

    >>> times = time + np.linspace(0 * u.min, 2 * u.min, 50)
    >>> orbit(times).shape
    (50,)

    """  # noqa: E501

    def __init__(self, line1, line2):
        self._tle = Satrec.twoline2rv(line1, line2)

    @classmethod
    def from_file(cls, name_or_obj):
        with get_readable_fileobj(name_or_obj) as f:
            *_, line1, line2 = f.readlines()
        return cls(line1, line2)

    @classmethod
    def from_id(cls, norad_id: int):
        """Get the latest TLE for a satellite from Celestrak.

        Examples
        --------

        Look up the latest TLE for BurstCube.

        >>> from m4opt.orbit import TLE
        >>> tle = TLE.from_id(59562)
        """
        *_, line1, line2 = fetch_tle_from_celestrak(norad_id)
        return cls(line1, line2)

    @property
    def period(self):
        """The orbital period at the epoch of the TLE."""
        return 2 * np.pi / self._tle.no * u.minute

    def __call__(self, time):
        """Get the position and velocity of the satellite.

        Parameters
        ----------
        time : :class:`astropy.time.Time`
            The time of the observation.

        Returns
        -------
        coord : :class:`astropy.coordinates.SkyCoord`
            The coordinates of the satellite in the ITRS frame.

        Notes
        -----
        The orbit propagation is based on the example code at
        https://docs.astropy.org/en/stable/coordinates/satellites.html.

        """
        shape = time.shape
        time = time.ravel()

        time = time.utc
        e, xyz, vxyz = self._tle.sgp4_array(time.jd1, time.jd2)
        x, y, z = xyz.T
        vx, vy, vz = vxyz.T

        # If any errors occurred, only raise for the first error
        e = e[e != 0]
        if e.size > 0:
            raise RuntimeError(SGP4_ERRORS[e[0]])

        coord = SkyCoord(
            x=x * u.km,
            v_x=vx * u.km / u.s,
            y=y * u.km,
            v_y=vy * u.km / u.s,
            z=z * u.km,
            v_z=vz * u.km / u.s,
            frame=TEME(obstime=time),
        ).itrs
        if shape:
            coord = coord.reshape(shape)
        else:
            coord = coord[0]
        return coord
