from functools import reduce
from operator import mul

import numpy as np
import sympy
from astropy import units as u
from astropy.modeling import CompoundModel, Model
from scipy.interpolate import interp1d
from synphot import SourceSpectrum, SpectralElement

from ._extrinsic import ScaleFactor, state
from .extinction._dust import DustExtinction, DustExtinctionForSkyCoord, dust_map


class ModelSymbol(sympy.Dummy):
    """A SymPy model to keep track of Astropy models in an expression."""

    def __new__(cls, model: Model):
        obj = super().__new__(cls, name=repr(model), real=True)
        obj.model = model
        return obj


def countrate(
    spectrum: SourceSpectrum, bandpass: SpectralElement
) -> u.Quantity[1 / (u.s * u.cm**2)]:
    """
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from m4opt.synphot.background import ZodiacalBackground
    >>> from m4opt.synphot.extinction import DustExtinction
    >>> from m4opt.synphot import observing
    >>> from m4opt.synphot._math import countrate
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
    <Quantity [[1.15881864e+14, 1.15881864e+14, 1.15881864e+14, ...,
                1.15881864e+14, 1.15881864e+14, 1.15881864e+14],
               [1.25522043e+14, 1.26524531e+14, 1.27836738e+14, ...,
                1.24963859e+14, 1.25203898e+14, 1.25522043e+14],
               [1.32664208e+14, 1.32527192e+14, 1.32032285e+14, ...,
                1.28400102e+14, 1.30881727e+14, 1.32664208e+14],
               ...,
               [9.94134356e+13, 8.77698007e+13, 9.46161206e+13, ...,
                1.13406002e+14, 1.09954677e+14, 9.94134356e+13],
               [1.22098526e+14, 1.19999232e+14, 1.17629623e+14, ...,
                1.17528190e+14, 1.20257836e+14, 1.22098526e+14],
               [1.05378588e+14, 1.05378588e+14, 1.05378588e+14, ...,
                1.05378588e+14, 1.05378588e+14, 1.05378588e+14]] 1 / (s cm2)>
    """
    count_rate_unit = 1 / (u.s * u.cm**2)
    scale_factors = []
    dust_extinction = None

    def model_to_expr(model):
        match model:
            case CompoundModel(op="+"):
                return model_to_expr(model.left) + model_to_expr(model.right)
            case CompoundModel(op="*"):
                return model_to_expr(model.left) * model_to_expr(model.right)
            case ScaleFactor():
                symbol = ModelSymbol(model)
                scale_factors.append(symbol)
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
        return (spectrum * bandpass).integrate(bandpass.waveset) / u.photon

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
        for coef, term in expr.expand().collect(scale_factors, evaluate=False).items()
    )
