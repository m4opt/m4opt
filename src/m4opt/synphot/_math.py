from functools import reduce
from operator import mul

import numpy as np
import sympy
from astropy import units as u
from astropy.modeling import CompoundModel, Model
from scipy.interpolate import RegularGridInterpolator, interp1d
from synphot import SourceSpectrum, SpectralElement

from ._extrinsic import ScaleFactor, state
from .extinction._atmosphere import AtmosphericExtinctionForSkyCoord
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
    <Quantity 1.12409479e+09 PHOTLAM>
    >>> with observing(EarthLocation.of_site('Palomar'), SkyCoord(*np.meshgrid(np.linspace(0, 360, 100), np.linspace(-90, 90, 200)), unit=u.deg), Time('2024-01-01')):
    ...     countrate(spectrum, band)
    <Quantity [[1.15140438e+14, 1.15140438e+14, 1.15140438e+14, ...,
                1.15140438e+14, 1.15140438e+14, 1.15140438e+14],
               [1.24826976e+14, 1.25836843e+14, 1.27159526e+14, ...,
                1.24264912e+14, 1.24506600e+14, 1.24826976e+14],
               [1.32033983e+14, 1.31895443e+14, 1.31395121e+14, ...,
                1.27727678e+14, 1.30232553e+14, 1.32033983e+14],
               ...,
               [9.86638417e+13, 8.70293885e+13, 9.38712128e+13, ...,
                1.12658842e+14, 1.09202848e+14, 9.86638417e+13],
               [1.21382128e+14, 1.19272528e+14, 1.16893607e+14, ...,
                1.16791828e+14, 1.19532294e+14, 1.21382128e+14],
               [1.04625376e+14, 1.04625376e+14, 1.04625376e+14, ...,
                1.04625376e+14, 1.04625376e+14, 1.04625376e+14]] 1 / (s cm2)>
    """
    count_rate_unit = 1 / (u.s * u.cm**2)
    scale_factors = []
    dust_extinction = None
    atmospheric_extinction = None

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
            case AtmosphericExtinctionForSkyCoord():
                symbol = ModelSymbol(model)
                nonlocal atmospheric_extinction
                atmospheric_extinction = symbol
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
        # Both kinds of extinction dim the source by an amount that depends on
        # where it is, through one number each: the reddening of the dust in
        # front of it, and the airmass of the line of sight. Rather than
        # integrate the spectrum separately for every target, integrate it over
        # a grid of those numbers and interpolate, which pays off as soon as
        # there are more targets than grid points.
        n_samples = 512

        @np.vectorize(otypes=[float])
        def countrate_for_Ebv(Ebv):
            return base_countrate_no_extinction(
                spectrum * DustExtinction(Ebv)
            ).to_value(count_rate_unit)

        @np.vectorize(otypes=[float])
        def countrate_for_airmass(airmass):
            return base_countrate_no_extinction(
                spectrum * atmospheric_extinction.model.at_airmass(airmass)
            ).to_value(count_rate_unit)

        @np.vectorize(otypes=[float])
        def countrate_for_both(Ebv, airmass):
            return base_countrate_no_extinction(
                spectrum
                * DustExtinction(Ebv)
                * atmospheric_extinction.model.at_airmass(airmass)
            ).to_value(count_rate_unit)

        def grid(values, size):
            """Points spanning the values, or None if they span nothing."""
            low, high = np.min(values), np.max(values)
            if not high > low:
                return None
            return np.linspace(low, high, size)

        if dust_extinction is not None:
            Ebv = dust_map().query(state.get().target_coord)
        if atmospheric_extinction is not None:
            airmass = atmospheric_extinction.model.airmass

        # Extinction is exponential in each of these, so the logarithm of the
        # count rate is nearly straight in them and interpolates far better
        # than the count rate itself.
        match (dust_extinction, atmospheric_extinction):
            case (None, None):
                return base_countrate_no_extinction(spectrum)
            case (_, None):
                x = None if np.size(Ebv) < n_samples else grid(Ebv, n_samples)
                if x is None:
                    return countrate_for_Ebv(Ebv) * count_rate_unit
                y = np.log(countrate_for_Ebv(x))
                return (
                    np.exp(
                        interp1d(x, y, kind="cubic", copy=False, assume_sorted=True)(
                            Ebv
                        )
                    )
                    * count_rate_unit
                )
            case (None, _):
                x = None if np.size(airmass) < n_samples else grid(airmass, n_samples)
                if x is None:
                    return countrate_for_airmass(airmass) * count_rate_unit
                y = np.log(countrate_for_airmass(x))
                return (
                    np.exp(
                        interp1d(x, y, kind="cubic", copy=False, assume_sorted=True)(
                            airmass
                        )
                    )
                    * count_rate_unit
                )
            case _:
                # Two parameters, so the grid is two-dimensional and each side
                # of it is correspondingly coarser.
                side = int(np.sqrt(n_samples))
                shape = np.broadcast_shapes(np.shape(Ebv), np.shape(airmass))
                x = y = None
                if np.prod(shape, dtype=int) >= side * side:
                    x, y = grid(Ebv, side), grid(airmass, side)
                if x is None or y is None:
                    return countrate_for_both(Ebv, airmass) * count_rate_unit
                z = np.log(countrate_for_both(x[:, np.newaxis], y[np.newaxis, :]))
                interp = RegularGridInterpolator((x, y), z, method="cubic")
                points = np.stack(np.broadcast_arrays(Ebv, airmass), axis=-1)
                return np.exp(interp(points)) * count_rate_unit

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
    # Each kind of extinction can be pulled out of the expression and applied
    # by interpolation instead. Any that will not factor out stays in the
    # spectrum and is integrated directly.
    if dust_extinction is not None:
        if (new_expr := expr.extract_multiplicatively(dust_extinction)) is not None:
            expr = new_expr
        else:
            dust_extinction = None
    if atmospheric_extinction is not None:
        if (
            new_expr := expr.extract_multiplicatively(atmospheric_extinction)
        ) is not None:
            expr = new_expr
        else:
            atmospheric_extinction = None

    return sum(
        evaluate_coef(coef) * evaluate_term(term)
        for coef, term in expr.expand().collect(scale_factors, evaluate=False).items()
    )
