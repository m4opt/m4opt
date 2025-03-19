"""Basic positional astronomy constraints."""

from abc import abstractmethod

from astropy import units as u
from astropy.coordinates import (
    ICRS,
    AltAz,
    Angle,
    EarthLocation,
    HADec,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.time import Time

from ..utils.typing_extensions import override
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
