from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz, get_sun

from ._core import Constraint


class AtNightConstraint(Constraint):
    """
    Constrain observations to times when the Sun is below a specified altitude.

    This constraint checks if the Sun's altitude is below a specified `solar_altitude_limit`.
    The threshold can be modified to represent various twilight stages.

    Parameters
    ----------
    max_solar_altitude : `~astropy.units.Quantity`, optional
        The altitude of the Sun below which it is considered "night" (inclusive).
        Default is `0 deg`.

    Examples
    --------

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import AtNightConstraint
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint_civil = AtNightConstraint.civil()
    >>> constraint_nautical = AtNightConstraint.nautical()
    >>> constraint_astronomical = AtNightConstraint.astronomical()
    >>> constraint_civil(location, None, time), constraint_nautical(location, None, time),   constraint_astronomical(location, None, time)
    (np.True_, np.True_, np.True_)
    """

    def __init__(self, solar_altitude_limit: Optional[u.Quantity] = 0 * u.deg):
        self.solar_altitude_limit = solar_altitude_limit

    def civil():
        return AtNightConstraint(solar_altitude_limit=-6 * u.deg)

    def nautical():
        return AtNightConstraint(solar_altitude_limit=-12 * u.deg)

    def astronomical():
        return AtNightConstraint(solar_altitude_limit=-18 * u.deg)

    def __call__(self, observer_location, target_coord, obstime):
        """
        Compute the nighttime constraint.
        """
        solar_altitude = (
            get_sun(obstime)
            .transform_to(
                AltAz(obstime=obstime, location=observer_location, pressure=0 * u.hPa)
            )
            .alt
        )
        return solar_altitude <= self.solar_altitude_limit
