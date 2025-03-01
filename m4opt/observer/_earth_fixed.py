from astropy.coordinates import EarthLocation

from ..utils.typing_extensions import override
from ._core import ObserverLocation


class EarthFixedObserverLocation(ObserverLocation, EarthLocation):
    """An observer at a fixed location on the surface of the Earth.

    >>> from astropy.time import Time
    >>> from m4opt.observer import EarthFixedObserverLocation
    >>> observer = EarthFixedObserverLocation.of_site("LSST")
    >>> observer(Time.now())
    <EarthFixedObserverLocation (1818939.00669747, -5208471.0353078, -3195171.4154367) m>
    """

    @override
    def __call__(self, _):
        return self
