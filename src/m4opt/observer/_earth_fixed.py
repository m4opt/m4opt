from dataclasses import dataclass
from typing import override

import numpy as np
from astropy.coordinates import EarthLocation

from ._core import ObserverLocation


@dataclass
class EarthFixedObserverLocation(ObserverLocation):
    """An observer at a fixed location on the surface of the Earth.

    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from m4opt.observer import EarthFixedObserverLocation
    >>> observer = EarthFixedObserverLocation(EarthLocation.of_site("LSST"))
    >>> observer(Time.now())
    <EarthLocation (1818939.00669747, -5208471.0353078, -3195171.4154367) m>
    """

    earth_location: EarthLocation
    """The time-independent Earth-fixed location of the observer."""

    @override
    def __call__(self, time):
        return np.broadcast_to(self.earth_location, time.shape, subok=True)
