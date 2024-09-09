import numpy as np
import numpy.typing as npt
import spiceypy as spice
from astropy import units as u
from astropy.coordinates import ITRS, SkyCoord
from astropy.time import Time
from astropy.utils.data import download_files_in_parallel

from ._core import Orbit


def _time_to_et(time: Time) -> float | npt.NDArray[np.floating]:
    """Convert an Astropy time to a SPICE elapsed time since epoch."""
    return (time.tdb - Time("J2000")).sec


# SPICE routines vectorized over time argument
_spkgps = np.vectorize(spice.spkgps, excluded=[0, 2, 3], signature="()->(m),()")


class Spice(Orbit):
    """An Earth satellite whose orbit is specified by its TLE.

    Parameters
    ----------
    bsp : str, file
        The filename or file-like object containing the Spice kernel.

    Examples
    --------

    Load an example Spice kernel from a file:

    >>> from importlib import resources
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.orbit import Spice
    >>> import numpy as np
    >>> orbit = Spice(
    ...     'MGS SIMULATION',
    ...     'https://archive.stsci.edu/missions/tess/models/TESS_EPH_PRE_LONG_2021252_21.bsp',
    ...     'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc',
    ...     'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc')
    >>> t0 = Time('2021-10-31 00:00')
    >>> orbit(t0)
    <SkyCoord (ITRS: obstime=2021-10-31 00:00:00.000, location=(0.0, 0.0, 0.0) km): (x, y, z) in km
        (259589.01504305, 267775.69181568, -6003.44398346)>
    >>> orbit(t0 + np.arange(4) * u.hour)
    <SkyCoord (ITRS: obstime=['2021-10-31 00:00:00.000' '2021-10-31 01:00:00.000'
     '2021-10-31 02:00:00.000' '2021-10-31 03:00:00.000'], location=(0.0, 0.0, 0.0) km): (x, y, z) in km
        [(259589.01504305, 267775.69181568, -6003.44398346),
         (319237.52597807, 193229.46894121, -5243.0107051 ),
         (358011.10059696, 105924.84848415, -4482.00355537),
         (373338.18149427,  11586.48011847, -3720.50003742)]>

    """  # noqa: E501

    def __init__(self, target: str, *kernels: str):
        if kernels:
            for filename in download_files_in_parallel(kernels, cache=True):
                spice.furnsh(filename)
        self._target = spice.bodn2c(target)
        self._body = spice.bodn2c("EARTH")

    def __call__(self, time):
        et = _time_to_et(time)
        pos, _ = _spkgps(self._target, et, "IAU_EARTH", self._body)
        return SkyCoord(*pos.T, unit=u.km, frame=ITRS(obstime=time))
