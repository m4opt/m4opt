"""Supernova SED models."""

import copy

import numpy as np
from astropy import units as u

from ._base import SEDModel
from ._lightcurves import VillarLightcurve
from ._parameters import PriorParameter
from ._priors import LogNormalPrior
from ._spectra import BlackbodySpectrum

__all__ = ("VillarCoolingBlackbodySED",)


class VillarCoolingBlackbodySED(SEDModel):
    """A supernova light curve with an evolving blackbody photosphere.

    Combines `VillarLightcurve`'s bolometric light curve with a
    time-dependent blackbody spectral shape,

    .. math::

        L_\\nu(\\nu, t) = L_\\mathrm{bol}(t)\\,S_\\mathrm{BB}[\\nu, T(t)]

    where the photospheric temperature cools as

    .. math::

        T(t) = T_\\mathrm{floor} + (T_0-T_\\mathrm{floor})(1+t/\\tau_T)^{-\\alpha_T}

    starting at :math:`T_0` and approaching :math:`T_\\mathrm{floor}`
    asymptotically. This is a generic phenomenological description of a
    cooling-photosphere transient (Villar et al. 2019); it does not model
    photospheric-radius evolution, line blanketing, or radiative transfer.

    Because temperature evolves with time, this is *not* a simple product
    of an independent light curve and spectrum (unlike
    `~m4opt.synphot.models.ComposedSEDModel`) -- the spectral shape itself
    depends on ``time`` through :math:`T(t)`.
    """

    amplitude = copy.deepcopy(VillarLightcurve.amplitude)
    t0 = copy.deepcopy(VillarLightcurve.t0)
    gamma = copy.deepcopy(VillarLightcurve.gamma)
    beta = copy.deepcopy(VillarLightcurve.beta)
    tau_rise = copy.deepcopy(VillarLightcurve.tau_rise)
    tau_fall = copy.deepcopy(VillarLightcurve.tau_fall)

    T0 = PriorParameter(
        default=1.2e4 * u.K, unit=u.K, prior=LogNormalPrior(np.log(1.2e4), 0.3, u.K)
    )
    T_floor = PriorParameter(
        default=6e3 * u.K, unit=u.K, prior=LogNormalPrior(np.log(6e3), 0.3, u.K)
    )
    tau_T = PriorParameter(
        default=15.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(15.0), 0.5, u.day),
    )
    alpha_T = PriorParameter(
        default=1.5 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(1.5), 0.3, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(
        freq,
        time,
        amplitude,
        t0,
        gamma,
        beta,
        tau_rise,
        tau_fall,
        T0,
        T_floor,
        tau_T,
        alpha_T,
    ):
        temperature = T_floor + (T0 - T_floor) * (1.0 + time / tau_T) ** (-alpha_T)
        bolometric = VillarLightcurve.evaluate(
            time, amplitude, t0, gamma, beta, tau_rise, tau_fall
        )
        shape = BlackbodySpectrum.evaluate(freq, temperature)
        return bolometric * shape
