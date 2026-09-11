from dataclasses import dataclass
from typing import override

from ..synphot.background._zodiacal import (
    ZodiacalBackgroundScaleFactor,
)
from ._core import Constraint

mag_at = ZodiacalBackgroundScaleFactor()._mag_at


@dataclass
class ZodiacalBackgroundConstraint(Constraint):
    r"""
    Constrain the surface brightness of the zodiacal light background.

    See Also
    --------
    m4opt.synphot.background.ZodiacalBackground

    Notes
    -----
    This uses the same model as
    :class:`m4opt.synphot.background.ZodiacalBackground`, and has all of the
    same limitations on its realm of validity.

    Examples
    --------
    .. plot::

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import EarthLocation, GeocentricTrueEcliptic, SkyCoord
        from astropy.time import Time
        from astropy.visualization import quantity_support
        from matplotlib import pyplot as plt

        from m4opt.constraints import ZodiacalBackgroundConstraint

        obstime = Time("2026-03-20T12:06:05.072")
        observer_location = EarthLocation.from_geocentric(0 * u.m, 0 * u.m, 0 * u.m)
        lon = np.linspace(-180, 180, 500) * u.deg
        lat = np.linspace(-90, 90, 500) * u.deg
        target_coord = SkyCoord(
            *np.meshgrid(lon, lat), frame=GeocentricTrueEcliptic(obstime=obstime)
        )
        in_constraint = np.zeros(target_coord.shape)
        delta = 0.25
        levels = np.arange(22, 23.25 + delta, delta)
        for surface_brightness in levels:
            constraint = ZodiacalBackgroundConstraint(surface_brightness)
            in_constraint[constraint(observer_location, target_coord, obstime)] = (
                surface_brightness
            )

        quantity_support()
        ax = plt.axes()
        ax.contour(
            lon,
            lat,
            in_constraint,
            levels=np.arange(levels.min() - 0.5 * delta, levels.max() + delta, delta),
            cmap="viridis_r",
        ).clabel(fmt=lambda value: f"{value + delta / 2:g}", colors="black")
        ax.xaxis.set_major_locator(plt.MultipleLocator(45))
        ax.yaxis.set_major_locator(plt.MultipleLocator(30))
        ax.set_xlabel(r"Ecliptic longitude relative to Sun, $\lambda - \lambda_\odot$")
        ax.set_ylabel(r"Ecliptic latitude, $\beta$")
        ax.grid()
    """

    surface_brightness: float
    r"""Most intense allowed visual surface brightness, :math:`m_V \, \mathrm{arcsec}^{-2}`."""

    @override
    def __call__(self, *args):
        return mag_at(*args) >= self.surface_brightness
