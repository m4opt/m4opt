r"""
Numerical helpers shared by the spectral shape models in :mod:`m4opt.models.spectra`.
"""

import numpy as np
from astropy import constants as const

from m4opt.models._typing import FloatArray

# ------------------------------------------ #
# Physical Constants (cgs)                   #
# ------------------------------------------ #
H_CGS: float = const.h.cgs.value
C_CGS: float = const.c.cgs.value
K_B_CGS: float = const.k_B.cgs.value
SIGMA_SB_CGS: float = const.sigma_sb.cgs.value


def log_expm1(x: FloatArray) -> FloatArray:
    r"""
    Compute :math:`\log(\exp(x) - 1)`, stable for both small and large ``x``.

    Used to evaluate the natural log of the Planck function's occupation
    factor :math:`(\exp(h\nu/k_BT) - 1)^{-1}` without overflowing for large
    arguments or losing precision for small ones.

    Parameters
    ----------
    x : array_like
        Input value(s). Assumed non-negative.

    Returns
    -------
    numpy.ndarray
        :math:`\log(\exp(x) - 1)`, using the asymptotic form :math:`\approx x`
        for ``x > 30`` (where :func:`numpy.expm1` would otherwise overflow).
    """
    x = np.asarray(x, dtype=np.float64)
    large = x > 30.0
    safe_x = np.where(large, 0.0, x)

    # `safe_x` is 0 on the `large` branch, purely to keep `expm1`/`log` from
    # overflowing there -- that branch's `log(expm1(0)) = -inf` is discarded
    # by `np.where` below, not a real result, so the warning it would
    # otherwise raise is suppressed.
    with np.errstate(divide="ignore"):
        return np.where(large, x, np.log(np.expm1(safe_x)))


def powerlaw_shape_integral_cgs(
    spectral_index: FloatArray,
    reference_frequency: FloatArray,
    frequency_min: FloatArray,
    frequency_max: FloatArray,
) -> FloatArray:
    r"""
    Integral of an unnormalized power-law shape over a finite frequency range.

    For a shape :math:`S(\nu) = \nu_0^{-1}(\nu/\nu_0)^\alpha`,

    .. math::

        \int_{\nu_\min}^{\nu_\max} S(\nu)\,d\nu =
        \begin{cases}
        \dfrac{x_{\max}^{\alpha+1} - x_{\min}^{\alpha+1}}{\alpha+1}, & \alpha \ne -1, \\
        \ln(x_{\max}/x_{\min}), & \alpha = -1,
        \end{cases}

    where :math:`x = \nu/\nu_0`. Broadcasts over its arguments -- used both
    for :class:`~m4opt.models.spectra.powerlaw.PowerLawSpectrum`'s own
    normalization and, applied twice with a shared reference frequency, for
    :class:`~m4opt.models.spectra.powerlaw.BrokenPowerLawSpectrum`'s two
    segments.

    Parameters
    ----------
    spectral_index : array_like
        Power-law index :math:`\alpha` in :math:`S(\nu) \propto \nu^\alpha`.
    reference_frequency : array_like
        Reference frequency :math:`\nu_0`, in Hz.
    frequency_min : array_like
        Lower integration limit, in Hz.
    frequency_max : array_like
        Upper integration limit, in Hz.

    Returns
    -------
    numpy.ndarray
        The dimensionless integral, broadcast over the input arrays.
    """
    alpha = np.asarray(spectral_index, dtype=np.float64)
    x_min = np.asarray(frequency_min, dtype=np.float64) / reference_frequency
    x_max = np.asarray(frequency_max, dtype=np.float64) / reference_frequency
    exponent = alpha + 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        power_law_integral = (x_max**exponent - x_min**exponent) / exponent
        log_integral = np.log(x_max / x_min)

    return np.where(np.isclose(exponent, 0.0), log_integral, power_law_integral)
