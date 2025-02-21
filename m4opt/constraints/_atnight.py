from typing import Optional

from astropy import units as u
from astropy.coordinates import AltAz, get_sun

from ..utils.typing_extensions import override
from ._core import Constraint


class AtNightConstraint(Constraint):
    """
    Constrain observations to times when the Sun is below a specified altitude.

    This constraint determines whether the Sun is below a given ``max_solar_altitude``.
    This altitude can be adjusted to define different twilight phases.

    Parameters
    ----------
    max_solar_altitude : `~astropy.units.Quantity`, optional
        The altitude of the Sun below which it is considered "night" (inclusive).
        Default is `0 deg`.
    force_pressure_zero : bool, optional
        If True, the pressure is set to zero when calculating the Sun's altitude.
        This avoids errors due to atmospheric refraction at very low altitudes.

    Examples
    --------
    To create a constraint that considers nighttime when the Sun is below -12° (nautical twilight):

        >>> from astropy.coordinates import EarthLocation, SkyCoord
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from m4opt.constraints import AtNightConstraint
        >>> time = Time("2017-08-17T00:41:04Z")
        >>> target = SkyCoord.from_name("NGC 4993")
        >>> location = EarthLocation.of_site("Rubin Observatory")
        >>> constraint = AtNightConstraint.twilight_civil()
        >>> constraint(location, None, time)
        np.True_
    """

    @u.quantity_input(horizon=u.deg)
    def __init__(
        self,
        max_solar_altitude: Optional[u.Quantity] = 0 * u.deg,
        force_pressure_zero: bool = True,
    ):
        self.max_solar_altitude = max_solar_altitude
        self.force_pressure_zero = force_pressure_zero

    @classmethod
    def twilight_civil(cls, **kwargs):
        """
        Consider nighttime as the period between civil twilights (-6 degrees).
        """
        return cls(max_solar_altitude=-6 * u.deg, **kwargs)

    @classmethod
    def twilight_nautical(cls, **kwargs):
        """
        Consider nighttime as the period between nautical twilights (-12 degrees).
        """
        return cls(max_solar_altitude=-12 * u.deg, **kwargs)

    @classmethod
    def twilight_astronomical(cls, **kwargs):
        """
        Consider nighttime as the period between astronomical twilights (-18 degrees).
        """
        return cls(max_solar_altitude=-18 * u.deg, **kwargs)

    @override
    def __call__(self, observer_location, target_coord, obstime):
        """
        Compute the nighttime constraint.

        Parameters
        ----------
        observer_location : `~astropy.coordinates.EarthLocation`
            The observing location.
        target_coord : `~astropy.coordinates.SkyCoord`, optional
            The celestial coordinates of the target (not used in this constraint).
        obstime : `~astropy.time.Time`
            The observation time.

        Returns
        -------
        `numpy.ndarray`
            Boolean mask indicating whether the Sun is below `max_solar_altitude`.
        """
        solar_altitude = self._get_solar_altitudes(obstime, observer_location)
        return solar_altitude <= self.max_solar_altitude

    def _get_solar_altitudes(self, obstime, observer_location):
        """
        Compute the altitude of the Sun at the given times and location.

        Parameters
        ----------
        obstime : `~astropy.time.Time`
            The observation time.
        observer_location  : `~astropy.coordinates.EarthLocation`
            The observer location.

        Returns
        -------
        `~astropy.units.Quantity`
            The altitude of the Sun.
        """

        altaz_frame = AltAz(
            obstime=obstime,
            location=observer_location,
            pressure=0 * u.hPa if self.force_pressure_zero else None,
        )
        solar_altitude = get_sun(obstime).transform_to(altaz_frame).alt

        return solar_altitude
