"""Basic positional astronomy constraints."""

from abc import abstractmethod

from astropy import units as u
from astropy.coordinates import (
    ICRS,
    AltAz,
    EarthLocation,
    HADec,
    UnitSphericalRepresentation,
)
from astropy.time import Time

from ..utils.typing_extensions import override
from ._core import Constraint


class CoordConstraint(Constraint):
    def __init__(
        self, min: u.Quantity[u.physical.angle], max: u.Quantity[u.physical.angle]
    ):
        self._min, self._max = sorted((min, max))

    @abstractmethod
    def _frame(self, observer_location: EarthLocation, obstime: Time):
        """Frame for this constraint"""

    @override
    def __call__(self, observer_location, target_coord, obstime):
        attr = getattr(
            target_coord.transform_to(
                self._frame(observer_location, obstime)
            ).represent_as(UnitSphericalRepresentation),
            self._key,
        )
        return (self._min <= attr) & (attr <= self._max)


class AltAzConstraint(CoordConstraint):
    @override
    def _frame(self, observer_location, obstime):
        return AltAz(obstime=obstime, location=observer_location)


class HADecConstraint(CoordConstraint):
    @override
    def _frame(self, observer_location, obstime):
        return HADec(obstime=obstime, location=observer_location)


class RaDecConstraint(CoordConstraint):
    @override
    def _frame(self, observer_location, obstime):
        return ICRS()


class LonConstraint(CoordConstraint):
    _key = "lon"

    def __init__(
        self,
        min: u.Quantity[u.physical.angle] = 0 * u.deg,
        max: u.Quantity[u.physical.angle] = 360 * u.deg,
    ):
        super().__init__(min, max)


class LatConstraint(CoordConstraint):
    _key = "lat"

    def __init__(
        self,
        min: u.Quantity[u.physical.angle] = -90 * u.deg,
        max: u.Quantity[u.physical.angle] = 90 * u.deg,
    ):
        super().__init__(min, max)


class AltitudeConstraint(LatConstraint, AltAzConstraint):
    """Constrain the altitude of the target.

    See Also
    --------
    AzimuthConstraint
    """


class AzimuthConstraint(LonConstraint, AltAzConstraint):
    """Constrain the azimuth of the target.

    See Also
    --------
    AltitudeConstraint
    """


class RightAscensionConstraint(LonConstraint, RaDecConstraint):
    """Constrain the ICRS right ascension of the target.

    See Also
    --------
    DeclinationConstraint
    """


class DeclinationConstraint(LatConstraint, RaDecConstraint):
    """Constrain the ICRS declination of the target.

    See Also
    --------
    RightAscensionConstraint
    """


class HourAngleConstraint(LonConstraint, HADecConstraint):
    """Constrain the hour angle of the target."""
