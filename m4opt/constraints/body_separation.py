from astropy import units as u
from astropy.coordinates import get_body

from .core import Constraint


class BodySeparationConstraint(Constraint):
    def __init__(self, min: u.Quantity[u.deg], body: str):
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
        self._body = body
        self.min = min

    def __call__(self, observer_location, target_coord, obstime):
        return (
            get_body(self._body, time=obstime, location=observer_location).separation(
                target_coord, origin_mismatch="ignore"
            )
            >= self.min
        )


class MoonSeparationConstraint(BodySeparationConstraint):
    _body = "moon"

    def __init__(self, min: u.Quantity[u.deg]):
        """
        Constrain the minimum separation from the Moon.

        Parameters
        ----------
        min : :class:`astropy.units.Quantity`
            Minimum angular separation from the Moon.

        Example
        -------

        >>> from astropy.coordinates import EarthLocation, SkyCoord
        >>> from astropy.time import Time
        >>> from astropy import units as u
        >>> from m4opt.constraints import MoonSeparationConstraint
        >>> time = Time("2017-08-17T12:41:04Z")
        >>> target = SkyCoord.from_name("NGC 4993")
        >>> location = EarthLocation.of_site("Las Campanas Observatory")
        >>> constraint = MoonSeparationConstraint(20 * u.deg)
        >>> constraint(location, target, time)
        np.True_
        """
        super().__init__(min, "moon")


class SunSeparationConstraint(BodySeparationConstraint):
    _body = "sun"

    def __init__(self, min: u.Quantity[u.deg]):
        """
        Constrain the minimum separation from the Sun.

        Parameters
        ----------
        min : :class:`astropy.units.Quantity`
            Minimum angular separation from the Sun.

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
        super().__init__(min, "sun")
