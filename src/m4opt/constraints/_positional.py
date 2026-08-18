"""Basic positional astronomy constraints."""

from abc import abstractmethod
from typing import override

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    ICRS,
    AltAz,
    Angle,
    EarthLocation,
    GeocentricTrueEcliptic,
    HADec,
    SkyCoord,
    UnitSphericalRepresentation,
    get_sun,
)
from astropy.time import Time

from ._core import Constraint


class AngleConstraint(Constraint):
    _key: str

    def __init__(
        self,
        min: u.Quantity[u.physical.angle] | Angle,
        max: u.Quantity[u.physical.angle] | Angle,
    ):
        self._min = min
        self._max = max

    @abstractmethod
    def _frame(self, observer_location: EarthLocation, obstime: Time):
        """Frame for this constraint"""

    def _get_angle(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ):
        return getattr(
            target_coord.transform_to(
                self._frame(observer_location, obstime)
            ).represent_as(UnitSphericalRepresentation),
            self._key,
        )

    @override
    def __call__(self, *args):
        angle = self._get_angle(*args)
        return (self._min <= angle) & (angle <= self._max)


class AltAzConstraint(AngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.AltAz` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return AltAz(obstime=obstime, location=observer_location)


class HADecConstraint(AngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.HADec` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return HADec(obstime=obstime, location=observer_location)


class GeocentricTrueEclipticConstraint(AngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.GeocentricTrueEclipticConstraint` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return GeocentricTrueEcliptic(obstime=obstime)


class ICRSConstraint(AngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.ICRS` frame."""

    @override
    def _frame(self, *_):
        return ICRS()


class LongitudeConstraint(AngleConstraint):
    """Constrain a generic longitude-like angle.

    Notes
    -----
    The allowed interval extends from the minimum angle to the maximum angle.
    For example, if the minimum and maximum angle are 10° and 30° respectively,
    then the constraint will return true over an interval of 20°. However, if
    the minimum and maximum angle are 30° and 10°, then the constraint will
    return true over an interval of 340°.
    """

    _key = "lon"

    @override
    def __init__(self, min, max):
        super().__init__(Angle(min).wrap_at(max), max)

    @override
    def _get_angle(self, *args):
        return super()._get_angle(*args).wrap_at(self._max)


class LatitudeConstraint(AngleConstraint):
    """Constrain a generic latitude-like angle.

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.
    """

    _key = "lat"

    @override
    def __init__(self, *args):
        super().__init__(*sorted(args))


class AltitudeConstraint(LatitudeConstraint, AltAzConstraint):
    """Constrain the altitude of the target.

    See Also
    --------
    AzimuthConstraint
    """


class AzimuthConstraint(LongitudeConstraint, AltAzConstraint):
    """Constrain the azimuth of the target.

    See Also
    --------
    AltitudeConstraint
    """


class RightAscensionConstraint(LongitudeConstraint, ICRSConstraint):
    """Constrain the ICRS right ascension of the target.

    See Also
    --------
    DeclinationConstraint
    """


class DeclinationConstraint(LatitudeConstraint, ICRSConstraint):
    """Constrain the ICRS declination of the target.

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.

    See Also
    --------
    RightAscensionConstraint
    """


class HourAngleConstraint(LongitudeConstraint, HADecConstraint):
    """Constrain the hour angle of the target.

    See Also
    --------
    RightAscensionConstraint
    """


class EclipticLatitudeConstraint(LatitudeConstraint, GeocentricTrueEclipticConstraint):
    """Constrain the ecliptic latitude of the target.

    This is the angle :math:`β` of Leinert et al. (1998), Fig. 2
    :footcite:`1998A&AS..127....1L`.

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.

    See Also
    --------
    HelioeclipticLongitudeConstraint, SunSeparationConstraint

    References
    ----------
    .. footbibliography::
    """


class HelioeclipticLongitudeConstraint(GeocentricTrueEclipticConstraint):
    """Constrain the helioecliptic longitude of the target.

    This places a constraint on the absolute value, between 0° and 180°, of the
    ecliptic longitude of the target minus the ecliptic longitude of the sun.
    This is the angle :math:`|λ - λ_⊙|` of Leinert et al. (1998), Fig. 2
    :footcite:`1998A&AS..127....1L`.

    See Also
    --------
    EclipticLatitudeConstraint, SunSeparationConstraint

    References
    ----------
    .. footbibliography::

    Warnings
    --------
    This model should only be used for observers near Earth --- in Earth orbit,
    as Hubble is, or on the Earth, or even on the Moon or in cislunar space. It
    should NOT be used for observers in orbits around other planets, or in
    distant solar orbits, or at Earth-Sun Lagrange points.
    """

    @override
    def _get_angle(self, observer_location, target_coord, obstime):
        frame = self._frame(observer_location, obstime)
        sun = get_sun(obstime)
        lon = (
            target_coord.transform_to(frame)
            .represent_as(UnitSphericalRepresentation)
            .lon
        )
        lon0 = sun.transform_to(frame).represent_as(UnitSphericalRepresentation).lon
        return np.abs((lon - lon0).wrap_at(180 * u.deg))
