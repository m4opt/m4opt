"""Spectral shapes: normalized functions of frequency alone.

Each returns a shape :math:`S(\\nu)` normalized so
:math:`\\int_0^\\infty S(\\nu)\\,d\\nu = 1`, in units of 1/Hz -- so that
multiplying by a bolometric `~m4opt.synphot.models.LightcurveModel` gives a
properly normalized :math:`L_\\nu(\\nu, t)`.
"""

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.modeling import Parameter

from ._base import Spectrum
from ._parameters import PriorParameter
from ._priors import LogNormalPrior, NormalPrior

__all__ = ("BlackbodySpectrum", "BrokenPowerLawSpectrum", "PowerLawSpectrum")


def _log_expm1(x):
    """log(exp(x) - 1), stable for large x (uses the x >> 1 asymptote exp(x))."""
    x = np.asarray(x, dtype=np.float64)
    large = x > 30.0
    safe_x = np.where(large, 0.0, x)
    with np.errstate(divide="ignore"):
        return np.where(large, x, np.log(np.expm1(safe_x)))


def _powerlaw_shape_integral(spectral_index, reference_frequency, freq_min, freq_max):
    """Integral of nu_0^-1 (nu/nu_0)^alpha over [freq_min, freq_max]."""
    alpha = np.asarray(spectral_index, dtype=np.float64)
    x_min = (freq_min / reference_frequency).to_value(u.dimensionless_unscaled)
    x_max = (freq_max / reference_frequency).to_value(u.dimensionless_unscaled)
    exponent = alpha + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        power_law_integral = (x_max**exponent - x_min**exponent) / exponent
        log_integral = np.log(x_max / x_min)
    return np.where(np.isclose(exponent, 0.0), log_integral, power_law_integral)


class BlackbodySpectrum(Spectrum):
    """A blackbody spectral shape, shaped entirely by ``temperature``.

    .. math::

        S(\\nu, T) = \\pi B_\\nu(\\nu, T) / (\\sigma_\\mathrm{SB} T^4)

    the Lambertian-emergent Planck function normalized by the
    Stefan-Boltzmann constant. Unlike a power law, this needs no reference
    frequency: it already integrates to 1 over all frequencies for any
    temperature.
    """

    temperature = PriorParameter(
        default=1e4 * u.K, unit=u.K, prior=LogNormalPrior(np.log(1e4), 0.5, u.K)
    )

    @staticmethod
    def evaluate(freq, temperature):
        x = (const.h * freq / (const.k_B * temperature)).to_value(
            u.dimensionless_unscaled
        )
        log_b_nu = (
            np.log(2.0 * const.h.cgs.value / const.c.cgs.value**2)
            + 3.0 * np.log(freq.to_value(u.Hz))
            - _log_expm1(x)
        )
        log_shape = (
            np.log(np.pi)
            + log_b_nu
            - np.log(const.sigma_sb.cgs.value)
            - 4.0 * np.log(temperature.to_value(u.K))
        )
        # log_shape was computed entirely from cgs-stripped values, so its
        # exponential is already the value of S(nu, T) in cgs units of 1/Hz.
        return np.exp(log_shape) / u.Hz


class PowerLawSpectrum(Spectrum):
    """A truncated power-law spectral shape.

    .. math::

        S(\\nu) \\propto \\nu^\\alpha, \\quad \\nu_\\min \\le \\nu \\le \\nu_\\max

    normalized to integrate to 1 over ``[frequency_min, frequency_max]``,
    zero outside it. ``reference_frequency`` just anchors the otherwise
    arbitrary normalization and cancels out once the shape is normalized.
    """

    spectral_index = PriorParameter(
        default=-1.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=NormalPrior(-1.0, 0.5, u.dimensionless_unscaled),
    )
    reference_frequency = Parameter(default=1e15 * u.Hz, unit=u.Hz, fixed=True)
    frequency_min = Parameter(default=1e12 * u.Hz, unit=u.Hz, fixed=True)
    frequency_max = Parameter(default=1e20 * u.Hz, unit=u.Hz, fixed=True)

    @staticmethod
    def evaluate(
        freq, spectral_index, reference_frequency, frequency_min, frequency_max
    ):
        valid = (freq >= frequency_min) & (freq <= frequency_max)
        safe_freq = np.where(valid, freq.to_value(u.Hz), reference_frequency.value)
        ratio = safe_freq / reference_frequency.to_value(u.Hz)
        integral = _powerlaw_shape_integral(
            spectral_index, reference_frequency, frequency_min, frequency_max
        )
        shape = ratio**spectral_index / reference_frequency.to_value(u.Hz) / integral
        return np.where(valid, shape, 0.0) / u.Hz


class BrokenPowerLawSpectrum(Spectrum):
    """A two-segment power-law spectral shape, continuous at ``break_frequency``.

    Uses index ``spectral_index_1`` below ``break_frequency`` and
    ``spectral_index_2`` above it, normalized to integrate to 1 over
    ``[frequency_min, frequency_max]``, zero outside it. Both segments share
    ``break_frequency`` as their reference frequency, so the shape is
    automatically continuous there.
    """

    spectral_index_1 = PriorParameter(
        default=-1.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=NormalPrior(-1.0, 0.5, u.dimensionless_unscaled),
    )
    spectral_index_2 = PriorParameter(
        default=-2.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=NormalPrior(-2.0, 0.5, u.dimensionless_unscaled),
    )
    break_frequency = Parameter(default=1e15 * u.Hz, unit=u.Hz, fixed=True)
    frequency_min = Parameter(default=1e12 * u.Hz, unit=u.Hz, fixed=True)
    frequency_max = Parameter(default=1e20 * u.Hz, unit=u.Hz, fixed=True)

    @staticmethod
    def evaluate(
        freq,
        spectral_index_1,
        spectral_index_2,
        break_frequency,
        frequency_min,
        frequency_max,
    ):
        valid = (freq >= frequency_min) & (freq <= frequency_max)
        safe_freq = np.where(valid, freq.to_value(u.Hz), break_frequency.value)
        break_freq_hz = break_frequency.to_value(u.Hz)
        alpha = np.where(safe_freq < break_freq_hz, spectral_index_1, spectral_index_2)
        integral = _powerlaw_shape_integral(
            spectral_index_1, break_frequency, frequency_min, break_frequency
        ) + _powerlaw_shape_integral(
            spectral_index_2, break_frequency, break_frequency, frequency_max
        )
        ratio = safe_freq / break_freq_hz
        shape = ratio**alpha / break_freq_hz / integral
        return np.where(valid, shape, 0.0) / u.Hz
