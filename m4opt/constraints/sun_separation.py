from astropy import units as u
from astropy.coordinates import get_body

from .core import Constraint


class SunSeparationConstraint(Constraint):
    def __init__(self, min: u.Quantity[u.deg]):
        """
        Example
        -------

        >>> from astropy.coordinates import EarthLocation, SkyCoord
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from m4opt.constraints import SunSeparationConstraint
        >>> time = Time("2017-08-17T12:41:04Z")
        >>> target = SkyCoord.from_name("NGC 4993")
        >>> location = EarthLocation.of_site("Las Campanas Observatory")
        >>> constraint = SunSeparationConstraint(20 * u.deg)
        >>> constraint(location, target, time)
        np.True_
        """
        self.min = min

    def __call__(self, observer_location, target_coord, obstime):
        return (
            get_body("sun", time=obstime, location=observer_location).separation(
                target_coord, origin_mismatch="ignore"
            )
            >= self.min
        )
