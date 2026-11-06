import json
import os
import time
import warnings
from collections.abc import Mapping
from typing import Any, override
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from astropy import units as u
from astropy.config import get_cache_dir_path
from astropy.coordinates import TEME, SkyCoord
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, accelerated

from ._core import ObserverLocation

if not accelerated:
    raise RuntimeError(
        "python-sgp4 was installed without the required accelerated API. Are you using a version of Python that is not yet supported by M4OPT? "
    )


_CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

# Celestrak checks for new GP data once every 2 hours and blocks clients that
# download more often. See
# https://celestrak.org/NORAD/documentation/gp-data-formats.php.
_CELESTRAK_UPDATE_INTERVAL = 2 * 60 * 60


def _fetch_omm_from_celestrak(norad_id: int) -> dict[str, Any]:
    """Get OMM fields for a satellite from Celestrak, with on-disk caching.

    Cached data is reused if it is less than 2 hours old. If a download fails
    and cached data exists, the cached data is used, with a warning, and the
    next download is not attempted for another 2 hours.
    """
    path = get_cache_dir_path() / "m4opt" / "celestrak" / f"{norad_id}.json"
    try:
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        age = None

    if age is None or age >= _CELESTRAK_UPDATE_INTERVAL:
        url = f"{_CELESTRAK_GP_URL}?{urlencode({'CATNR': norad_id, 'FORMAT': 'json'})}"
        try:
            with urlopen(url, timeout=60) as response:
                content = response.read()
            (fields,) = json.loads(content)
        except (HTTPError, URLError, ValueError) as e:
            if age is None:
                raise RuntimeError(
                    f"Failed to get orbital elements for NORAD ID {norad_id} from {url}: {e}"
                ) from e
            warnings.warn(
                f"Failed to update orbital elements for NORAD ID {norad_id} "
                f"from {url}: {e}. Using cached data from {path}."
            )
            os.utime(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
            tmp_path.write_bytes(content)
            tmp_path.replace(path)
            return fields

    (fields,) = json.loads(path.read_bytes())
    return fields


class TleObserverLocation(ObserverLocation):
    """
    An Earth satellite whose orbit is specified by its `two-line element (TLE) <https://www.space-track.org/documentation#tle>`_.

    Notes
    -----
    The orbit propagation is based on the example code at
    :ref:`astropy-coordinates-satellites`.

    Examples
    --------

    Here is an example TLE for BurstCube (NORAD ID 59562):

    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> import numpy as np
    >>> from m4opt.observer import TleObserverLocation
    >>> line1 = '1 59562U 98067WM  24220.55604657  .00200610  00000+0  13802-2 0  9999'
    >>> line2 = '2 59562  51.6321  52.1851 0005233 205.4266 154.6476 15.73479335 17375'
    >>> orbit = TleObserverLocation(line1, line2)

    Evaluate the position and velocity of the satellite at one specific time:

    >>> time = Time('2024-08-08 01:10:41')
    >>> orbit(time)
    <EarthLocation (4172.83539674, -514.29983197, -5254.79814474) km>

    Or evaluate at an array of times:

    >>> times = time + np.linspace(0 * u.min, 2 * u.min, 50)
    >>> orbit(times).shape
    (50,)

    If any orbit propagation errors occur, then a :class:`RuntimeError` is
    raised:

    >>> time = Time('2025-01-01 00:00:00')
    >>> orbit(time)
    Traceback (most recent call last):
      ...
    RuntimeError: mrt is less than 1.0 which indicates the satellite has decayed
    """

    _tle: Satrec

    def __init__(self, line1: str, line2: str):
        """Create a TLE from the text of its two lines."""
        self._tle = Satrec.twoline2rv(line1, line2)

    @classmethod
    def from_omm(cls, fields: Mapping[str, Any]) -> "TleObserverLocation":
        """
        Create an orbit from Orbit Mean-Elements Message (OMM) fields.

        See `Celestrak's documentation
        <https://celestrak.org/NORAD/documentation/gp-data-formats.php>`_ for
        a description of the OMM format. Unlike TLEs, OMMs support NORAD catalog numbers greater than 99999.

        Examples
        --------

        Here are the OMM fields for BurstCube (NORAD ID 59562), equivalent to
        the TLE in the class example above:

        >>> from astropy.time import Time
        >>> from m4opt.observer import TleObserverLocation
        >>> fields = {
        ...     "OBJECT_NAME": "BURSTCUBE",
        ...     "OBJECT_ID": "1998-067WM",
        ...     "EPOCH": "2024-08-07T13:20:42.423648",
        ...     "MEAN_MOTION": 15.73479335,
        ...     "ECCENTRICITY": 0.0005233,
        ...     "INCLINATION": 51.6321,
        ...     "RA_OF_ASC_NODE": 52.1851,
        ...     "ARG_OF_PERICENTER": 205.4266,
        ...     "MEAN_ANOMALY": 154.6476,
        ...     "EPHEMERIS_TYPE": 0,
        ...     "CLASSIFICATION_TYPE": "U",
        ...     "NORAD_CAT_ID": 59562,
        ...     "ELEMENT_SET_NO": 999,
        ...     "REV_AT_EPOCH": 1737,
        ...     "BSTAR": 0.0013802,
        ...     "MEAN_MOTION_DOT": 0.0020061,
        ...     "MEAN_MOTION_DDOT": 0,
        ... }
        >>> orbit = TleObserverLocation.from_omm(fields)
        >>> orbit(Time('2024-08-08 01:10:41'))
        <EarthLocation (4172.83539674, -514.29983197, -5254.79814474) km>
        """
        self = cls.__new__(cls)
        self._tle = Satrec()
        omm.initialize(self._tle, fields)
        return self

    @classmethod
    def from_id(cls, norad_id: int) -> "TleObserverLocation":
        """
        Get the latest orbital elements for a satellite from Celestrak.

        The elements are downloaded in OMM format, so this works for NORAD
        catalog numbers greater than 99999. Downloads are cached and reused
        for up to 2 hours, as requested by Celestrak's `usage policy
        <https://celestrak.org/NORAD/documentation/gp-data-formats.php>`_. If
        Celestrak cannot be reached, then previously cached elements are used.

        Examples
        --------

        Look up the latest orbital elements for the Fermi Gamma-Ray Space
        Telescope.

        >>> from m4opt.observer import TleObserverLocation
        >>> tle = TleObserverLocation.from_id(33053)  # doctest: +SKIP
        """
        return cls.from_omm(_fetch_omm_from_celestrak(norad_id))

    @override
    def __call__(self, time):
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
        ).itrs.earth_location
        if shape:
            coord = coord.reshape(shape)
        else:
            coord = coord[0]
        return coord
