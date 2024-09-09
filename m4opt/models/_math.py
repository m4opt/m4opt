from functools import reduce
from operator import mul

import numpy as np
import sympy
from astropy import units as u
from astropy.modeling import CompoundModel, Model
from scipy.interpolate import interp1d
from synphot import Observation, SourceSpectrum, SpectralElement

from ._extinction import DustExtinction, DustExtinctionForSkyCoord, dust_map
from ._extrinsic import ExtrinsicScaleFactor, state


class ModelSymbol(sympy.Dummy):
    """A SymPy model to keep track of Astropy models in an expression."""

    def __new__(cls, model: Model):
        obj = super().__new__(cls, name=repr(model), real=True)
        obj.model = model
        return obj


def countrate(
    spectrum: SourceSpectrum, bandpass: SpectralElement
) -> u.Quantity[u.count / (u.s * u.cm**2)]:
    """
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.models.background import ZodiacalBackground
    >>> from m4opt.models import DustExtinction
    >>> from m4opt.models import observing
    >>> from m4opt.models._math import countrate
    >>> import numpy as np
    >>> import synphot
    >>> from astropy import units as u
    >>> spectrum = synphot.SourceSpectrum(synphot.BlackBody1D, temperature=1000 * u.Kelvin) * DustExtinction()
    >>> band = synphot.SpectralElement.from_filter('johnson_r')
    >>> with observing(EarthLocation.of_site('Palomar'), SkyCoord(0 * u.deg, 0 * u.deg), Time('2024-01-01')):
    ...     (spectrum * band)(6000 * u.angstrom)
    <Quantity 1.12557248e+09 PHOTLAM>
    >>> with observing(EarthLocation.of_site('Palomar'), SkyCoord(*np.meshgrid(np.linspace(0, 360, 100), np.linspace(-90, 90, 200)), unit=u.deg), Time('2024-01-01')):
    ...     countrate(spectrum, band)
    <Quantity [[1.16887524e+14, 1.16887524e+14, 1.16887524e+14, ...,
                1.16887524e+14, 1.16887524e+14, 1.16887524e+14],
               [1.26595837e+14, 1.27605355e+14, 1.28926750e+14, ...,
                1.26033735e+14, 1.26275459e+14, 1.26595837e+14],
               [1.33787878e+14, 1.33649910e+14, 1.33151561e+14, ...,
                1.29494053e+14, 1.31992994e+14, 1.33787878e+14],
               ...,
               [1.00300151e+14, 8.85700541e+13, 9.54674777e+13, ...,
                1.14393997e+14, 1.10917927e+14, 1.00300151e+14],
               [1.23148237e+14, 1.21034114e+14, 1.18647710e+14, ...,
                1.18545557e+14, 1.21294547e+14, 1.23148237e+14],
               [1.06308797e+14, 1.06308797e+14, 1.06308797e+14, ...,
                1.06308797e+14, 1.06308797e+14, 1.06308797e+14]] ct / (s cm2)>
    """
    count_rate_unit = u.count / (u.s * u.cm**2)
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

    def base_countrate_no_extinction(spectrum):
        area = 1 * u.cm**2
        return (
            Observation(spectrum, bandpass).countrate(
                wavelengths=bandpass.waveset,
                area=area,
            )
            / area
        )

    def base_countrate(spectrum):
        @np.vectorize
        def base_countrate_extinction_for_Ebv(Ebv):
            return base_countrate_no_extinction(
                spectrum * DustExtinction(Ebv)
            ).to_value(count_rate_unit)

        if dust_extinction is not None:
            xp = dust_map().query(state.get().target_coord)
            n_samples = 512
            if np.size(xp) >= n_samples:
                x = np.linspace(0, dust_extinction.model.Ebv_max, n_samples)
                y = base_countrate_extinction_for_Ebv(x)
                xp = dust_map().query(state.get().target_coord)
                return (
                    interp1d(x, y, kind="cubic", copy=False, assume_sorted=True)(xp)
                    * count_rate_unit
                )
            else:
                return base_countrate_extinction_for_Ebv(xp) * count_rate_unit
        else:
            return base_countrate_no_extinction(spectrum)

    def evaluate_term(term):
        match term:
            case sympy.Add():
                return sum(evaluate_term(arg) for arg in term.args)
            case sympy.Mul():
                return base_countrate(
                    SourceSpectrum(reduce(mul, (arg.model for arg in term.args)))
                )
            case sympy.Symbol():
                return base_countrate(SourceSpectrum(term.model))
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
