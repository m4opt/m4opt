import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from ..utils.typing_extensions import override
from ._core import Constraint


def _get_angle_from_earth_limb(
    observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
) -> u.Quantity[u.physical.angle]:
    alt = target_coord.transform_to(
        AltAz(location=observer_location, obstime=obstime)
    ).alt
    x, y, z = observer_location.geocentric
    z = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    with np.errstate(invalid="ignore"):
        limb_alt = np.arccos(R_earth / z)
    return alt + limb_alt


class EarthLimbConstraint(Constraint):
    """
    Constrain the angle from the Earth limb.

    Parameters
    ----------
    min
        Minimum angular separation from the Earth's limb.

    Notes
    -----
    This constraint assumes a spherical Earth, so it is only accurate to about
    a degree for observers in very low Earth orbit (height of 100 km).

    Examples
    --------

    >>> from astropy.constants import R_earth
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import EarthLimbConstraint
    >>> observer_location = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 2 * R_earth)
    >>> target_coord = SkyCoord(300 * u.deg, -30 * u.deg)
    >>> obstime = Time.now()
    >>> constraint = EarthLimbConstraint(10 * u.deg)
    >>> constraint(observer_location, target_coord, obstime)
    np.True_
    """

    def __init__(self, min: u.Quantity[u.physical.angle]):
        self.min = min

    @override
    def __call__(self, *args):
        return _get_angle_from_earth_limb(*args) >= self.min
