r"""
Blackbody spectral shape.

A :class:`BlackbodySpectrum` is shaped entirely by a single sampled
``temperature`` -- unlike a power law, which is scale-free and needs an
arbitrary reference frequency to anchor it, a blackbody's shape has no free
normalization choice: the Planck function, normalized by the
Stefan-Boltzmann law, already integrates to exactly 1 over all frequencies
for any temperature.
"""

from typing import ClassVar

import numpy as np
from astropy import units as u

from m4opt.models._typing import CGSParameterValue, FloatArray
from m4opt.models.core._base import Spectrum
from m4opt.models.core._parameters import Parameter
from m4opt.models.core.priors import LogNormalPrior

from ._utils import C_CGS, H_CGS, K_B_CGS, SIGMA_SB_CGS, log_expm1

__all__ = ["BlackbodySpectrum"]


class BlackbodySpectrum(Spectrum):
    r"""
    Blackbody spectral shape, shaped entirely by a sampled ``temperature``.

    Represents :math:`S(\nu, T) = \pi B_\nu(\nu, T) / (\sigma T^4)`, the
    Lambertian-emergent Planck function normalized by the Stefan-Boltzmann
    constant so that :math:`\int_0^\infty S(\nu, T)\,d\nu = 1` for any
    ``temperature`` -- see the module docstring for why this needs no
    reference-frequency pivot the way a power law does.

    By default ``temperature`` uses a
    :class:`~m4opt.models.core.priors.LogNormalPrior`, a physically motivated
    prior for a strictly positive scale parameter; like any other
    :class:`~m4opt.models.core._parameters.Parameter`, it can be overridden
    per instance.

    .. rubric:: Parameters

    The spectral shape parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``temperature``
         - :math:`T`
         - Blackbody temperature.
    """

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "temperature": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e4 * u.K,
            description="Blackbody temperature.",
            latex=r"T",
        ),
    }

    # ----------------------------------- #
    # Shape: S(nu)                        #
    # ----------------------------------- #
    @classmethod
    def _eval(cls, nu: FloatArray, *, temperature: CGSParameterValue) -> FloatArray:
        x = H_CGS * nu / (K_B_CGS * temperature)
        log_B_nu = np.log(2.0 * H_CGS / C_CGS**2) + 3.0 * np.log(nu) - log_expm1(x)

        return (
            np.log(np.pi) + log_B_nu - np.log(SIGMA_SB_CGS) - 4.0 * np.log(temperature)
        )

    # ----------------------------------- #
    # Normalization: integral of S(nu) dnu #
    # ----------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Natural log of the shape integral.

        Exactly 0 (i.e. the integral itself is 1) for any ``temperature``, by
        construction -- see the class docstring.
        """
        return np.zeros_like(np.asarray(parameters["temperature"], dtype=np.float64))
