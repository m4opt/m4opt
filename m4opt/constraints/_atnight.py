from astropy import units as u
from astropy.coordinates import AltAz, get_sun

from ..utils.typing_extensions import override
from ._core import Constraint


class AtNightConstraint(Constraint):
    """
     Constrain observations to specific twilight phases or a user-defined solar altitude limit.

    Parameters
    ----------
    max_solar_altitude : `~astropy.units.Quantity`
        A user-defined maximum solar altitude threshold. This parameter is required
        when not using specific twilight methods (`twilight_civil`, `twilight_nautical`, `twilight_astronomical`).
        It sets the maximum altitude of the Sun for which observations are allowed.

    Notes
    -----
    - The pressure is set to zero when calculating the Sun's altitude.
      This avoids errors due to atmospheric refraction at very low altitudes.

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import AtNightConstraint
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint = AtNightConstraint.twilight_civil()
    >>> constraint(observer_location=location, target_coord=target, obstime=time)
    np.True_
    >>> constraint = AtNightConstraint.twilight_nautical()
    >>> constraint(observer_location=location, target_coord=target, obstime=time)
    np.True_
    >>> constraint = AtNightConstraint.twilight_astronomical()
    >>> constraint(observer_location=location, target_coord=target, obstime=time)
    np.True_
    """

    def __init__(self, max_solar_altitude: u.Quantity[u.physical.angle] = 0 * u.deg):
        self.max_solar_altitude = max_solar_altitude

    @classmethod
    def twilight_civil(cls):
        """
        Create an :class:`~m4opt.constraints.AtNightConstraint` for civil twilight (-6°).
        """
        return cls(-6 * u.deg)

    @classmethod
    def twilight_nautical(cls):
        """
        Create an :class:`~m4opt.constraints.AtNightConstraint` for nautical twilight (-12°).
        """
        return cls(-12 * u.deg)

    @classmethod
    def twilight_astronomical(cls):
        """
        Create an :class:`~m4opt.constraints.AtNightConstraint` for astronomical twilight (-18°).
        """
        return cls(-18 * u.deg)

    @override
    def __call__(self, observer_location, target_coord, obstime):
        altaz_frame = AltAz(obstime=obstime, location=observer_location)
        sun_altitude = get_sun(obstime).transform_to(altaz_frame).alt
        return sun_altitude <= self.max_solar_altitude
