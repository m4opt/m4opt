import numpy as np
import numpy.typing as npt
import spiceypy as spice
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils.data import download_file

from ..utils.typing_extensions import override
from ._core import ObserverLocation


def _time_to_et(time: Time) -> float | npt.NDArray[np.floating]:
    """Convert an Astropy time to a SPICE elapsed time since epoch."""
    return (time.tdb - Time("J2000")).sec


# SPICE routines vectorized over time argument
_spkgps = np.vectorize(spice.spkgps, excluded=[0, 2, 3], signature="()->(m),()")


class SpiceObserverLocation(ObserverLocation):
    """A satellite whose orbit is specified by `Spice <https://naif.jpl.nasa.gov/naif/>`_ kernels.

    Examples
    --------

    Load an example Spice kernel from a file:

    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.observer import SpiceObserverLocation
    >>> import numpy as np
    >>> orbit = SpiceObserverLocation(
    ...     'MGS SIMULATION',
    ...     'https://archive.stsci.edu/missions/tess/models/TESS_EPH_PRE_LONG_2021252_21.bsp',
    ...     'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc',
    ...     'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc')
    >>> t0 = Time('2021-10-31 00:00')
    >>> orbit(t0)
    <EarthLocation (259589.01504305, 267775.69181568, -6003.44398346) km>
    >>> orbit(t0 + np.arange(4) * u.hour).shape
    (4,)
    """  # noqa: E501

    def __init__(self, target: str, *kernels: str):
        for kernel in kernels:
            spice.furnsh(download_file(kernel, cache=True))
        self._target = spice.bodn2c(target)
        self._body = spice.bodn2c("EARTH")

    @override
    def __call__(self, time):
        et = _time_to_et(time)
        pos, _ = _spkgps(self._target, et, "IAU_EARTH", self._body)
        return EarthLocation.from_geocentric(*pos.T, unit=u.km)
