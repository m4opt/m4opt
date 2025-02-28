from astropy.coordinates import EarthLocation

from ..utils.typing_extensions import override
from ._core import Orbit


class EarthFixed(Orbit, EarthLocation):
    """An observer at a fixed location on the surface of the Earth.

    >>> from astropy.time import Time
    >>> from m4opt.orbit import EarthFixed
    >>> observer = EarthFixed.of_site("LSST")
    >>> observer(Time.now())
    <EarthFixed (1818939.00669747, -5208471.0353078, -3195171.4154367) m>
    """

    @override
    def __call__(self, _):
        return self
