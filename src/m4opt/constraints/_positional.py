"""Basic positional astronomy constraints."""

from abc import abstractmethod
from dataclasses import dataclass
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

from ..dynamics._roll import nominal_roll
from ._core import Constraint


@dataclass
class AngleConstraint(Constraint):
    min: u.Quantity[u.physical.angle] | Angle
    """Minimum angle."""

    max: u.Quantity[u.physical.angle] | Angle
    """Maximum angle."""

    @abstractmethod
    def _get_angle(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ) -> u.Quantity[u.physical.angle] | Angle:
        """Calculate the angle that is bounded by this constraint."""
        raise NotImplementedError

    @override
    def __call__(self, *args):
        angle = self._get_angle(*args)
        return (self.min <= angle) & (angle <= self.max)


class WrappedAngleConstraint(AngleConstraint):
    def __post_init__(self):
        self.min = Angle(self.min).wrap_at(self.max)

    @override
    def __call__(self, *args):
        angle = Angle(self._get_angle(*args)).wrap_at(self.max)
        return (self.min <= angle) & (angle <= self.max)


class FrameAngleConstraint(AngleConstraint):
    _key: str

    @abstractmethod
    def _frame(self, observer_location: EarthLocation, obstime: Time):
        """Frame for this constraint."""
        raise NotImplementedError

    @override
    def _get_angle(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ):
        return getattr(
            target_coord.transform_to(
                self._frame(observer_location, obstime)
            ).represent_as(UnitSphericalRepresentation),
            self._key,
        )


class AltAzConstraint(FrameAngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.AltAz` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return AltAz(obstime=obstime, location=observer_location)


class HADecConstraint(FrameAngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.HADec` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return HADec(obstime=obstime, location=observer_location)


class GeocentricTrueEclipticConstraint(FrameAngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.GeocentricTrueEclipticConstraint` frame."""

    @override
    def _frame(self, observer_location, obstime):
        return GeocentricTrueEcliptic(obstime=obstime)


class ICRSConstraint(FrameAngleConstraint):
    """Constrain an angle in the :class:`~astropy.coordinates.ICRS` frame."""

    @override
    def _frame(self, *_):
        return ICRS()


@dataclass
class LongitudeConstraint(FrameAngleConstraint, WrappedAngleConstraint):
    """
    Constrain a generic longitude-like angle.

    Notes
    -----
    The allowed interval extends from the minimum angle to the maximum angle.
    For example, if the minimum and maximum angle are 10° and 30° respectively,
    then the constraint will return true over an interval of 20°. However, if
    the minimum and maximum angle are 30° and 10°, then the constraint will
    return true over an interval of 340°.
    """

    _key = "lon"


class LatitudeConstraint(FrameAngleConstraint):
    """
    Constrain a generic latitude-like angle.

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.
    """

    _key = "lat"

    @override
    def __init__(self, *args):
        super().__init__(*sorted(args))


class AltitudeConstraint(LatitudeConstraint, AltAzConstraint):
    """
    Constrain the altitude of the target.

    See Also
    --------
    AzimuthConstraint
    """


class AzimuthConstraint(LongitudeConstraint, AltAzConstraint):
    """
    Constrain the azimuth of the target.

    See Also
    --------
    AltitudeConstraint
    """


class RightAscensionConstraint(LongitudeConstraint, ICRSConstraint):
    """
    Constrain the ICRS right ascension of the target.

    See Also
    --------
    DeclinationConstraint
    """


class DeclinationConstraint(LatitudeConstraint, ICRSConstraint):
    """
    Constrain the ICRS declination of the target.

    See Also
    --------
    RightAscensionConstraint

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.
    """


class HourAngleConstraint(LongitudeConstraint, HADecConstraint):
    """
    Constrain the hour angle of the target.

    See Also
    --------
    RightAscensionConstraint
    """


class EclipticLatitudeConstraint(LatitudeConstraint, GeocentricTrueEclipticConstraint):
    """
    Constrain the ecliptic latitude of the target.

    This is the angle :math:`β` of Leinert et al. (1998), Fig. 2
    :footcite:`1998A&AS..127....1L`.

    See Also
    --------
    HelioeclipticLongitudeConstraint, SunSeparationConstraint

    Notes
    -----
    If the maximum angle is less than the minimum angle, then they are swapped.

    References
    ----------
    .. footbibliography::
    """


class HelioeclipticLongitudeConstraint(GeocentricTrueEclipticConstraint):
    """
    Constrain the helioecliptic longitude of the target.

    This places a constraint on the absolute value, between 0° and 180°, of the
    ecliptic longitude of the target minus the ecliptic longitude of the sun.
    This is the angle :math:`|λ - λ_⊙|` of Leinert et al. (1998), Fig. 2
    :footcite:`1998A&AS..127....1L`.

    Warnings
    --------
    This model should only be used for observers near Earth --- in Earth orbit,
    as Hubble is, or on the Earth, or even on the Moon or in cislunar space. It
    should NOT be used for observers in orbits around other planets, or in
    distant solar orbits, or at Earth-Sun Lagrange points.

    See Also
    --------
    EclipticLatitudeConstraint, SunSeparationConstraint

    References
    ----------
    .. footbibliography::
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


@dataclass
class NominalRollConstraint(WrappedAngleConstraint):
    """
    Constrain the nominal roll angle due to solar panel constraints.

    See Also
    --------
    m4opt.dynamics.nominal_roll

    Examples
    --------
    .. plot::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time
        from matplotlib import pyplot as plt

        from m4opt.constraints import NominalRollConstraint
        from m4opt.dynamics import nominal_roll

        observer_location = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
        target_coord = SkyCoord.from_name('LMC')
        obstime = Time('2026-01-01') + np.linspace(0, 1, 180, endpoint=False) * u.year
        rolls = nominal_roll(observer_location, target_coord, obstime)

        def plot_roll_constraint(min, max, symmetry):
            constraint = NominalRollConstraint(min=min, max=max, symmetry=symmetry)
            keep = constraint(observer_location, target_coord, obstime)

            fig_width, _ = plt.rcParams['figure.figsize']
            fig = plt.figure(figsize=(fig_width, fig_width))
            ax = plt.axes(aspect=1)
            fig.suptitle(f'{min=:latex} {max=:latex} {symmetry=}')
            angles = rolls[keep].to_value(u.deg)
            x = np.zeros(len(angles))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.quiver(x, x, 1, 1, angles=angles, scale=3)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()

        plot_roll_constraint(20 * u.deg, 30 * u.deg, 4)
        plot_roll_constraint(170 * u.deg, -170 * u.deg, 1)
        plot_roll_constraint(180 * u.deg, 90 * u.deg, 1)
    """

    symmetry: int = 1
    """
    Rotational symmetry.

    For example, for the value 2, the either the roll angle or 180° plus the
    roll angle must be within the given limits.
    """

    @override
    def _get_angle(self, *args, **kwargs):
        step = np.linspace(0, 360, self.symmetry, endpoint=False) * u.deg
        return nominal_roll(*args, **kwargs)[..., np.newaxis] + step

    @override
    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs).any(axis=-1)
