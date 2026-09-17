"""Tidal disruption event SED models."""

import numpy as np
from astropy import units as u

from ._base import ComposedSEDModel
from ._lightcurves import GREDLightcurve
from ._parameters import PriorParameter
from ._priors import LogNormalPrior
from ._spectra import BlackbodySpectrum

__all__ = ("VanVelzenTDESED",)

_L = u.erg / u.s
_LN10 = np.log(10.0)


class _VanVelzenLightcurve(GREDLightcurve):
    """`GREDLightcurve` with the van Velzen et al. (2021) ZTF TDE population priors."""

    amplitude = PriorParameter(
        default=10**44 * _L,
        unit=_L,
        prior=LogNormalPrior(44.0 * _LN10, 0.2 * _LN10, _L),
    )
    sigma_rise = PriorParameter(
        default=10**1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(1.0 * _LN10, 0.22 * _LN10, u.day),
    )
    tau_decline = PriorParameter(
        default=10**1.7 * u.day,
        unit=u.day,
        prior=LogNormalPrior(1.7 * _LN10, 0.2 * _LN10, u.day),
    )


class _VanVelzenSpectrum(BlackbodySpectrum):
    """`BlackbodySpectrum` with the van Velzen et al. (2021) ZTF TDE population prior."""

    temperature = PriorParameter(
        default=10**4.3 * u.K,
        unit=u.K,
        prior=LogNormalPrior(4.3 * _LN10, 0.12 * _LN10, u.K),
    )


class VanVelzenTDESED(ComposedSEDModel):
    """A constant-temperature blackbody modulated by a Gaussian-rise/exponential-decay light curve.

    .. math::

        L_\\nu(\\nu, t) = L_\\mathrm{bol}(t)\\,S_\\mathrm{BB}(\\nu, T)

    Composed from `GREDLightcurve` (rise width :math:`\\sigma`, decline
    timescale :math:`\\tau`, peaking at :math:`L_0` after :math:`5\\sigma`)
    and `BlackbodySpectrum` at a constant photospheric temperature :math:`T`
    -- unlike `~m4opt.synphot.models.VillarCoolingBlackbodySED`, the
    temperature here does not evolve with time, so the light curve and
    spectral shape are independent and this is a valid `ComposedSEDModel`.
    """

    _LIGHTCURVE = _VanVelzenLightcurve
    _SPECTRUM = _VanVelzenSpectrum
