from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz, get_sun

from ..utils.typing_extensions import override
from ._core import Constraint


class TwilightConstraint(Constraint):
    def __init__(self, max_solar_altitude: Optional[u.Quantity] = 0 * u.deg):
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
    Constrain observations to times when the Sun is below a specified altitude.

    This constraint ensures that observations occur only when the Sun is below
    a given threshold altitude, which can be adjusted to match different twilight phases.

    Parameters
    ----------
    twilight_type : str, optional
        The type of twilight to use as the night threshold.
        Options: 'twilight_civil', 'twilight_nautical', 'twilight_astronomical'.
    max_solar_altitude : `~astropy.units.Quantity`, optional
        A custom maximum solar altitude threshold. If provided, it overrides the
        twilight type setting.

    Notes
    -----
    The pressure is set to zero when calculating the Sun's altitude.
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
    >>> constraint = AtNightConstraint("twilight_nautical")
    >>> constraint(observer_location=location, target_coord=target, obstime=time)
    np.True_
    """

    twilight_levels = {
        "twilight_civil": -6 * u.deg,
        "twilight_nautical": -12 * u.deg,
        "twilight_astronomical": -18 * u.deg,
    }

    def __init__(
        self,
        twilight_type: Optional[str] = None,
        max_solar_altitude: Optional[u.Quantity] = 0 * u.deg,
    ):
        if not isinstance(max_solar_altitude, u.Quantity):
            raise TypeError(
                f"max_solar_altitude must be an astropy Quantity with angular units, got {type(max_solar_altitude)} instead."
            )

        if twilight_type is not None and twilight_type not in self.twilight_levels:
            raise ValueError(
                f"Unknown twilight type: {twilight_type}. Choose from {list(self.twilight_levels.keys())}"
            )

        if max_solar_altitude.value == 0:
            max_solar_altitude = self.twilight_levels[twilight_type]

        super().__init__(max_solar_altitude=max_solar_altitude)

    @classmethod
    def twilight_civil(cls):
        """Returns an instance where night starts at civil twilight (-6°)."""
        return cls("twilight_civil")

    @classmethod
    def twilight_nautical(cls):
        """Returns an instance where night starts at nautical twilight (-12°)."""
        return cls("twilight_nautical")

    @classmethod
    def twilight_astronomical(cls):
        """Returns an instance where night starts at astronomical twilight (-18°)."""
        return cls("twilight_astronomical")
