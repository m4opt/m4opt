from functools import reduce
from operator import mul

import numpy as np
import sympy
from astropy.modeling import CompoundModel, Model
from scipy.interpolate import interp1d
from synphot import Observation, SourceSpectrum

from ._extinction import DustExtinction, DustExtinctionForSkyCoord, dust_map
from ._extrinsic import ExtrinsicScaleFactor, state


class ModelSymbol(sympy.Dummy):
    """A SymPy model to keep track of Astropy models in an expression."""

    def __new__(cls, model: Model):
        obj = super().__new__(cls, name=repr(model), real=True)
        obj.model = model
        return obj


def effstim(
    spectrum, bandpass, flux_unit=None, wavelengths=None, area=None, vegaspec=None
):
    """
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.models.background import ZodiacalBackground
    >>> from m4opt.models import DustExtinction
    >>> from m4opt.models import observing
    >>> from m4opt.models._math import effstim
    >>> import numpy as np
    >>> import synphot
    >>> from astropy import units as u
    >>> spectrum = synphot.SourceSpectrum(synphot.BlackBody1D, temperature=1000 * u.Kelvin) * synphot.SpectralElement(DustExtinction())
    >>> band = synphot.SpectralElement.from_filter('johnson_r')
    >>> with observing(EarthLocation.of_site('Palomar'), SkyCoord(0 * u.deg, 0 * u.deg), Time('2024-01-01')):
    ...     (spectrum * band)(2000 * u.angstrom)
    <Quantity 6.25485842e-18 PHOTLAM>
    >>> with observing(EarthLocation.of_site('Palomar'), SkyCoord(*np.meshgrid(np.linspace(0, 360, 100), np.linspace(-90, 90, 200)), unit=u.deg), Time('2024-01-01')):
    ...     effstim(spectrum, band)
    <Quantity [[5.56548638e+10, 5.56548638e+10, 5.56548638e+10, ...,
                5.56548638e+10, 5.56548638e+10, 5.56548638e+10],
               [6.02847758e+10, 6.07662430e+10, 6.13964602e+10, ...,
                6.00166956e+10, 6.01319796e+10, 6.02847758e+10],
               [6.37149606e+10, 6.36491559e+10, 6.34114654e+10, ...,
                6.16670282e+10, 6.28588841e+10, 6.37149606e+10],
               ...,
               [4.77455317e+10, 4.21534149e+10, 4.54415136e+10, ...,
                5.44657755e+10, 5.28081993e+10, 4.77455317e+10],
               [5.86405552e+10, 5.76323220e+10, 5.64942641e+10, ...,
                5.64455489e+10, 5.77565225e+10, 5.86405552e+10],
               [5.06104299e+10, 5.06104299e+10, 5.06104299e+10, ...,
                5.06104299e+10, 5.06104299e+10, 5.06104299e+10]] PHOTLAM>
    """

    if flux_unit is None:
        flux_unit = Observation._internal_flux_unit
    flux_unit = Observation._validate_flux_unit(flux_unit)

    extrinsic_scale_factors = []
    dust_extinction = None

    def model_to_expr(model):
        match model:
            case CompoundModel(op="+"):
                return model_to_expr(model.left) + model_to_expr(model.right)
            case CompoundModel(op="*"):
                return model_to_expr(model.left) * model_to_expr(model.right)
            case ExtrinsicScaleFactor():
                symbol = ModelSymbol(model)
                extrinsic_scale_factors.append(symbol)
                return symbol
            case DustExtinctionForSkyCoord():
                symbol = ModelSymbol(model)
                nonlocal dust_extinction
                dust_extinction = symbol
                return symbol
            case _:
                return ModelSymbol(model)

    def evaluate_coef(coef):
        match coef:
            case sympy.core.numbers.One():
                return 1
            case sympy.Symbol():
                return coef.model.value
            case _:
                raise NotImplementedError(
                    f"Don't know how to evaluate coefficient symbol: {coef}"
                )

    def base_effstim_no_extinction(spectrum):
        return Observation(spectrum, bandpass).effstim(
            flux_unit=flux_unit,
            wavelengths=bandpass.waveset,
            area=area,
            vegaspec=vegaspec,
        )

    def base_effstim(spectrum):
        @np.vectorize
        def base_effstim_extinction_for_Ebv(Ebv):
            return base_effstim_no_extinction(spectrum * DustExtinction(Ebv)).to_value(
                flux_unit
            )

        if dust_extinction is not None:
            xp = dust_map().query(state.get().target_coord)
            n_samples = 512
            if np.size(xp) >= n_samples:
                x = np.linspace(0, dust_extinction.model.Ebv_max, n_samples)
                y = base_effstim_extinction_for_Ebv(x)
                xp = dust_map().query(state.get().target_coord)
                return (
                    interp1d(x, y, kind="cubic", copy=False, assume_sorted=True)(xp)
                    * flux_unit
                )
            else:
                return base_effstim_extinction_for_Ebv(xp) * flux_unit
        else:
            return base_effstim_no_extinction(spectrum)

    def evaluate_term(term):
        match term:
            case sympy.Add():
                return sum(evaluate_term(arg) for arg in term.args)
            case sympy.Mul():
                return base_effstim(
                    SourceSpectrum(reduce(mul, (arg.model for arg in term.args)))
                )
            case sympy.Symbol():
                return base_effstim(SourceSpectrum(term.model))
            case _:
                raise NotImplementedError(
                    f"Don't know how to evaluate term symbol: {term}"
                )

    expr = model_to_expr(spectrum.model)
    if (
        dust_extinction is not None
        and (new_expr := expr.extract_multiplicatively(dust_extinction)) is not None
    ):
        expr = new_expr
    else:
        dust_extinction = None

    return sum(
        evaluate_coef(coef) * evaluate_term(term)
        for coef, term in expr.expand()
        .collect(extrinsic_scale_factors, evaluate=False)
        .items()
    )
