r"""
Composite tidal disruption event SED, following :footcite:t:`2021ApJ...908....4V`.
"""

from typing import ClassVar

from astropy import units as u

from m4opt.models.core._base import ComposedSpectralModel
from m4opt.models.core._parameters import Parameter
from m4opt.models.core.priors import NormalPrior
from m4opt.models.lightcurves.generic import GREDLightcurve
from m4opt.models.spectra.thermal import BlackbodySpectrum

__all__ = ["VanVelzenTDESED"]


class VanVelzenTDESED(ComposedSpectralModel):
    r"""
    A constant-temperature blackbody modulated by a Gaussian-rise, exponential-decay light curve.

    .. math::

        L_\nu(\nu, t) = L_0 \cdot \ell(t) \cdot \frac{\pi B_\nu(\nu, T)}{\sigma_\mathrm{SB} T^4},

    where

    .. math::

        \ell(t) = \begin{cases}
            \exp\left[-\dfrac{(t-t_\mathrm{peak})^2}{2\sigma^2}\right] & t < t_\mathrm{peak} \\[6pt]
            \exp\left[-\dfrac{t-t_\mathrm{peak}}{\tau}\right] & t \ge t_\mathrm{peak}
            \end{cases}

    with :math:`t_\mathrm{peak} = 5\sigma` (:class:`~m4opt.models.lightcurves.generic.GREDLightcurve`'s
    own convention -- the rise is always exactly 5 Gaussian widths long, not a
    separate free parameter). This is a fairly typical parameterization for a
    TDE SED :footcite:p:`2021ApJ...908....4V`: a constant-temperature
    blackbody photosphere (the :math:`\pi B_\nu(\nu, T)/(\sigma_\mathrm{SB}
    T^4)` factor is exactly :class:`~m4opt.models.spectra.thermal.BlackbodySpectrum`'s
    normalized shape, :math:`\int_0^\infty S(\nu, T)\,d\nu = 1`), so
    :math:`L_0` here is literally :math:`L_\mathrm{bol}(t_\mathrm{peak})`.
    Composed from :class:`~m4opt.models.lightcurves.generic.GREDLightcurve`
    (already bolometric -- :math:`L_0 \cdot \ell(t)`) and
    :class:`~m4opt.models.spectra.thermal.BlackbodySpectrum`; see
    :class:`~m4opt.models.core._base.ComposedSpectralModel` for how the two
    are combined.

    The default priors are the log-normal fits to the ZTF TDE sample
    reported by :footcite:t:`2021ApJ...908....4V` (their Section 4.1 /
    Table 4): log-normal in :math:`L_0`, :math:`T`, :math:`\sigma`, and
    :math:`\tau`, each parameterized here via a base-10 log transform on a
    :class:`~m4opt.models.core.priors.NormalPrior`.

    .. rubric:: Parameters

    The model parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`L_0`
         - Peak bolometric luminosity, L_0 = L_bol(t_peak). log10(L_0/[erg/s])
           ~ N(44, 0.2^2).
       * - ``sigma_rise``
         - :math:`\sigma`
         - Gaussian width of the pre-peak rise. log10(sigma/day) ~ N(1.0,
           0.22^2).
       * - ``tau_decline``
         - :math:`\tau`
         - Exponential decline timescale after peak. log10(tau/day) ~
           N(1.7, 0.2^2).
       * - ``temperature``
         - :math:`T`
         - Photospheric blackbody temperature. log10(T/K) ~ N(4.3, 0.12^2).

    References
    ----------
    .. footbibliography::
    """

    _LIGHTCURVE_CLASS = GREDLightcurve
    _SPECTRUM_CLASS = BlackbodySpectrum
    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=NormalPrior(mean=44.0, sigma=0.2),
            scale=1.0 * u.erg / u.s,
            transform="log10",
            description="Peak bolometric luminosity, L_0 = L_bol(t_peak). log10(L_0/[erg/s]) ~ N(44, 0.2^2).",
            latex=r"L_0",
        ),
        "temperature": Parameter(
            prior=NormalPrior(mean=4.3, sigma=0.12),
            scale=1.0 * u.K,
            transform="log10",
            description="Photospheric blackbody temperature. log10(T/K) ~ N(4.3, 0.12^2).",
            latex=r"T",
        ),
        "sigma_rise": Parameter(
            prior=NormalPrior(mean=1.0, sigma=0.22),
            scale=1.0 * u.day,
            transform="log10",
            description="Gaussian width of the pre-peak rise. log10(sigma/day) ~ N(1.0, 0.22^2).",
            latex=r"\sigma",
        ),
        "tau_decline": Parameter(
            prior=NormalPrior(mean=1.7, sigma=0.2),
            scale=1.0 * u.day,
            transform="log10",
            description="Exponential decline timescale after peak. log10(tau/day) ~ N(1.7, 0.2^2).",
            latex=r"\tau",
        ),
    }
