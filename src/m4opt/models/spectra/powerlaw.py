r"""
Power-law spectral shapes.

:class:`PowerLawSpectrum` represents a single truncated power law

.. math::

    S(\nu) = \frac{1}{\nu_0}\left(\frac{\nu}{\nu_0}\right)^\alpha,

between finite lower and upper frequency cutoffs and zero elsewhere. The
reference frequency :math:`\nu_0` fixes the otherwise arbitrary dimensional
normalization of the power law, while the finite cutoffs make the
frequency integral well-defined -- a pure, unbroken power law cannot have a
finite integral over :math:`0 < \nu < \infty`, since at least one end always
diverges.

:class:`BrokenPowerLawSpectrum` generalizes this to two power-law segments
that meet continuously at a ``break_frequency``, which doubles as the
reference frequency for both segments (so no separate normalization
constant is needed to keep the two pieces continuous).

Neither shape is normalized to integrate to 1 by construction; each
overrides :meth:`~m4opt.models.core._base.Spectrum._eval_normalization`
with the closed-form integral of its own shape (see
:func:`~m4opt.models.spectra._utils.powerlaw_shape_integral_cgs`).
"""

from typing import ClassVar

import numpy as np
from astropy import units as u

from m4opt.models._typing import CGSParameterValue, FloatArray
from m4opt.models.core._base import Spectrum
from m4opt.models.core._parameters import Parameter
from m4opt.models.core.priors import ConstantPrior, NormalPrior

from ._utils import powerlaw_shape_integral_cgs

__all__ = ["BrokenPowerLawSpectrum", "PowerLawSpectrum"]


class PowerLawSpectrum(Spectrum):
    r"""
    Finite-band power-law spectral shape.

    Represents :math:`S(\nu) = \nu_0^{-1}(\nu/\nu_0)^\alpha` between
    ``frequency_min`` and ``frequency_max``, and zero outside that interval.
    ``spectral_index`` is the usual frequency-space index :math:`\alpha`
    defined by :math:`S(\nu) \propto \nu^\alpha`.

    The default ``spectral_index`` is sampled from a Gaussian prior centered
    on -1. ``reference_frequency`` and the cutoffs use constant priors by
    default, making them deterministic unless replaced or overridden
    explicitly.

    .. rubric:: Parameters

    The spectral shape parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``spectral_index``
         - :math:`\alpha`
         - Frequency-space power-law index, S(nu) proportional to nu^alpha.
       * - ``reference_frequency``
         - :math:`\nu_0`
         - Reference frequency anchoring the power-law shape.
       * - ``frequency_min``
         - :math:`\nu_{\min}`
         - Lower frequency cutoff of the power-law spectrum.
       * - ``frequency_max``
         - :math:`\nu_{\max}`
         - Upper frequency cutoff of the power-law spectrum.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "spectral_index": Parameter(
            prior=NormalPrior(mean=-1.0, sigma=0.5),
            scale=1.0,
            description="Frequency-space power-law index, S(nu) proportional to nu^alpha.",
            latex=r"\alpha",
        ),
        "reference_frequency": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e15 * u.Hz,
            description="Reference frequency anchoring the power-law shape.",
            latex=r"\nu_0",
        ),
        "frequency_min": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e12 * u.Hz,
            description="Lower frequency cutoff of the power-law spectrum.",
            latex=r"\nu_{\min}",
        ),
        "frequency_max": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e20 * u.Hz,
            description="Upper frequency cutoff of the power-law spectrum.",
            latex=r"\nu_{\max}",
        ),
    }

    # ----------------------------------- #
    # Shape: S(nu)                        #
    # ----------------------------------- #
    @classmethod
    def _eval(
        cls,
        nu: FloatArray,
        *,
        spectral_index: CGSParameterValue,
        reference_frequency: CGSParameterValue,
        frequency_min: CGSParameterValue,
        frequency_max: CGSParameterValue,
    ) -> FloatArray:
        valid = (nu >= frequency_min) & (nu <= frequency_max)
        safe_nu = np.where(valid, nu, reference_frequency)

        log_shape = spectral_index * (
            np.log(safe_nu) - np.log(reference_frequency)
        ) - np.log(reference_frequency)

        return np.where(valid, log_shape, -np.inf)

    # ----------------------------------- #
    # Normalization: integral of S(nu) dnu #
    # ----------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        """Natural log of the analytic power-law integral over ``[frequency_min, frequency_max]``."""
        return np.log(
            powerlaw_shape_integral_cgs(
                parameters["spectral_index"],
                parameters["reference_frequency"],
                parameters["frequency_min"],
                parameters["frequency_max"],
            )
        )


class BrokenPowerLawSpectrum(Spectrum):
    r"""
    Finite-band, two-segment power-law spectral shape.

    Represents

    .. math::

        S(\nu) = \nu_{\rm break}^{-1}
        \left(\frac{\nu}{\nu_{\rm break}}\right)^{\alpha_1
        \,\mathrm{or}\,\alpha_2},

    using index :math:`\alpha_1` below ``break_frequency`` and
    :math:`\alpha_2` above it, and zero outside
    ``[frequency_min, frequency_max]``. Because both segments are anchored
    to the same ``break_frequency``, :math:`S(\nu)` is automatically
    continuous there (both give :math:`\nu_{\rm break}^{-1}`) without any
    extra matching condition.

    The default indices are sampled from Gaussian priors centered on -1
    (below the break) and -2 (above it) -- a generic steepening spectrum;
    like any other parameter, both can be overridden per instance.
    ``break_frequency`` and the cutoffs use constant priors by default,
    making them deterministic unless replaced or overridden explicitly.

    .. rubric:: Parameters

    The spectral shape parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``spectral_index_1``
         - :math:`\alpha_1`
         - Power-law index below ``break_frequency``.
       * - ``spectral_index_2``
         - :math:`\alpha_2`
         - Power-law index above ``break_frequency``.
       * - ``break_frequency``
         - :math:`\nu_{\rm break}`
         - Frequency at which the spectral index switches from alpha_1 to
           alpha_2.
       * - ``frequency_min``
         - :math:`\nu_{\min}`
         - Lower frequency cutoff of the spectrum.
       * - ``frequency_max``
         - :math:`\nu_{\max}`
         - Upper frequency cutoff of the spectrum.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "spectral_index_1": Parameter(
            prior=NormalPrior(mean=-1.0, sigma=0.5),
            scale=1.0,
            description="Power-law index below ``break_frequency``.",
            latex=r"\alpha_1",
        ),
        "spectral_index_2": Parameter(
            prior=NormalPrior(mean=-2.0, sigma=0.5),
            scale=1.0,
            description="Power-law index above ``break_frequency``.",
            latex=r"\alpha_2",
        ),
        "break_frequency": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e15 * u.Hz,
            description="Frequency at which the spectral index switches from alpha_1 to alpha_2.",
            latex=r"\nu_{\rm break}",
        ),
        "frequency_min": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e12 * u.Hz,
            description="Lower frequency cutoff of the spectrum.",
            latex=r"\nu_{\min}",
        ),
        "frequency_max": Parameter(
            prior=ConstantPrior(value=1.0),
            scale=1e20 * u.Hz,
            description="Upper frequency cutoff of the spectrum.",
            latex=r"\nu_{\max}",
        ),
    }

    # ----------------------------------- #
    # Shape: S(nu)                        #
    # ----------------------------------- #
    @classmethod
    def _eval(
        cls,
        nu: FloatArray,
        *,
        spectral_index_1: CGSParameterValue,
        spectral_index_2: CGSParameterValue,
        break_frequency: CGSParameterValue,
        frequency_min: CGSParameterValue,
        frequency_max: CGSParameterValue,
    ) -> FloatArray:
        valid = (nu >= frequency_min) & (nu <= frequency_max)
        safe_nu = np.where(valid, nu, break_frequency)
        alpha = np.where(safe_nu < break_frequency, spectral_index_1, spectral_index_2)

        log_shape = alpha * (np.log(safe_nu) - np.log(break_frequency)) - np.log(
            break_frequency
        )

        return np.where(valid, log_shape, -np.inf)

    # ----------------------------------- #
    # Normalization: integral of S(nu) dnu #
    # ----------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        """Natural log of the analytic integral, summed over the two segments."""
        break_frequency = parameters["break_frequency"]

        integral = powerlaw_shape_integral_cgs(
            parameters["spectral_index_1"],
            break_frequency,
            parameters["frequency_min"],
            break_frequency,
        ) + powerlaw_shape_integral_cgs(
            parameters["spectral_index_2"],
            break_frequency,
            break_frequency,
            parameters["frequency_max"],
        )

        return np.log(integral)
