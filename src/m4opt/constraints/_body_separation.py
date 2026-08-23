from typing import override

from astropy import units as u
from astropy.coordinates import get_body

from ._core import Constraint


class BodySeparationConstraint(Constraint):
    def __init__(self, min: u.Quantity[u.physical.angle], body: str):
        self._body = body
        self.min = min

    def _separation(self, observer_location, target_coord, obstime):
        return get_body(
            self._body, time=obstime, location=observer_location
        ).separation(target_coord, origin_mismatch="ignore")

    @override
    def __call__(self, observer_location, target_coord, obstime):
        return self._separation(observer_location, target_coord, obstime) >= self.min


class MoonSeparationConstraint(BodySeparationConstraint):
    """
    Constrain the minimum separation from the Moon.

    Parameters
    ----------
    min
        Minimum angular separation from the Moon.

    Examples
    --------

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

    def __init__(self, min: u.Quantity[u.physical.angle]):
        super().__init__(min, "moon")


class SunSeparationConstraint(BodySeparationConstraint):
    """
    Constrain the minimum separation from the Sun.

    This is the solar elongation :math:`ε` of Leinert et al. (1998), Fig. 2
    :footcite:`1998A&AS..127....1L`.

    Parameters
    ----------
    min
        Minimum angular separation from the Sun.

    See Also
    --------
    m4opt.constraints.EclipticLatitudeConstraint
    m4opt.constraints.HelioeclipticLongitudeConstraint

    References
    ----------
    .. footbibliography::

    Examples
    --------

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

    def __init__(self, min: u.Quantity[u.physical.angle]):
        super().__init__(min, "sun")


class AntiSolarSeparationConstraint(SunSeparationConstraint):
    """
    Constrain the minimum separation from the anti-solar point.

    The anti-solar point is the point on the sky directly opposite the
    Sun. The nominal spacecraft roll angle
    (see :func:`~m4opt.dynamics.nominal_roll`) is undefined there, and
    changes arbitrarily fast nearby.

    Parameters
    ----------
    min
        Minimum angular separation from the anti-solar point.

    See Also
    --------
    m4opt.constraints.SunSeparationConstraint
    m4opt.dynamics.nominal_roll

    Examples
    --------

    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import AntiSolarSeparationConstraint
    >>> time = Time("2017-08-17T12:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Las Campanas Observatory")
    >>> constraint = AntiSolarSeparationConstraint(20 * u.deg)
    >>> constraint(location, target, time)
    np.True_
    """

    @override
    def _separation(self, observer_location, target_coord, obstime):
        return 180 * u.deg - super()._separation(
            observer_location, target_coord, obstime
        )
