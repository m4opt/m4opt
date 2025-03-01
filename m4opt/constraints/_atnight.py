from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz, get_sun

from ..utils.typing_extensions import override
from ._core import Constraint


class TwilightConstraint(Constraint):
    """
    A base constraint to limit observations based on the Sun's altitude.

    This class is designed to be used as a parent class for specific twilight
    constraints, such as `AtNightConstraint`. It allows setting a maximum solar
    altitude to determine when observations are permissible.

    Notes
    -----
    - The pressure is set to zero when calculating the Sun's altitude.
      This avoids errors due to atmospheric refraction at very low altitudes.

    """

    def __init__(self, max_solar_altitude: Optional[u.Quantity]):
        if not isinstance(max_solar_altitude, u.Quantity):
            raise TypeError(
                f"max_solar_altitude must be an astropy Quantity with angular units, "
                f"got {type(max_solar_altitude)} instead."
            )
        self.max_solar_altitude = max_solar_altitude

    @override
    def __call__(self, observer_location, target_coord, obstime):
        altaz_frame = AltAz(
            obstime=obstime, location=observer_location, pressure=0 * u.hPa
        )
        sun_altitude = get_sun(obstime).transform_to(altaz_frame).alt
        return sun_altitude <= self.max_solar_altitude


class AtNightConstraint(TwilightConstraint):
    """
    Constrain observations to specific twilight phases based on the altitude of the Sun.

    Parameters
    ----------
    max_solar_altitude : `~astropy.units.Quantity`
        A user-defined maximum solar altitude threshold. This parameter is required
        when not using specific twilight methods (`twilight_civil`, `twilight_nautical`, `twilight_astronomical`).
        It sets the maximum altitude of the Sun for which observations are allowed.

    Notes
    -----
    - The twilight phase options correspond to the following solar altitudes:
      * 'twilight_civil': -6 degrees
      * 'twilight_nautical': -12 degrees
      * 'twilight_astronomical': -18 degrees

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
    """

    twilight_levels = {
        "twilight_civil": -6 * u.deg,
        "twilight_nautical": -12 * u.deg,
        "twilight_astronomical": -18 * u.deg,
    }

    def __init__(self, max_solar_altitude: u.Quantity):
        super().__init__(max_solar_altitude=max_solar_altitude)

    @classmethod
    def twilight_civil(cls):
        """
        Create an AtNightConstraint for civil twilight (-6°).
        """
        return cls(cls.twilight_levels["twilight_civil"])

    @classmethod
    def twilight_nautical(cls):
        """
        Create an AtNightConstraint for nautical twilight (-12°).
        """
        return cls(cls.twilight_levels["twilight_nautical"])

    @classmethod
    def twilight_astronomical(cls):
        """
        Create an AtNightConstraint for astronomical twilight (-18°).
        """
        return cls(cls.twilight_levels["twilight_astronomical"])
