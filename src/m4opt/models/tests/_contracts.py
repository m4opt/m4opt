"""
Shared, inheritable pytest "contracts" for :mod:`m4opt.models`.

Rather than writing bespoke tests for every :class:`~m4opt.models.core.Lightcurve`/
:class:`~m4opt.models.core.Spectrum`/:class:`~m4opt.models.core.SpectralModel`
subclass, each concrete model gets its own two-line ``Test*`` class that sets
``model_class`` and inherits from one of the ``*Contract`` mixins below. The
mixin supplies a battery of generic checks: instantiation, the ``Mapping``/
parameter-override contract shared by every model
(:class:`ModelContract`), and, per model kind, the ``eval*``-family
consistency invariants described in :mod:`m4opt.models.core._base` plus
normalization/bolometric-integral checks against independent numerical
integration.

None of the ``*Contract`` classes are collected by pytest directly -- they
are missing ``model_class`` and are not named ``Test*`` -- only their
concrete subclasses in ``test_lightcurves.py``/``test_spectra.py``/
``test_seds.py`` are.

To cover a new model, add one two-line class to the appropriate test module;
the completeness check at the bottom of that module will fail loudly if a
new model is added upstream without a corresponding test class.
"""

from __future__ import annotations

import copy
from collections.abc import Set as AbstractSet
from typing import ClassVar

import numpy as np
import pytest
from astropy import units as u
from astropy.units import Quantity
from scipy.integrate import quad

from m4opt.models._utils import AB_MAG_ZERO_POINT, to_cgs_value
from m4opt.models.core._parameters import Parameter

__all__ = [
    "LightcurveContract",
    "ModelContract",
    "SpectralModelContract",
    "SpectrumContract",
    "all_concrete_subclasses",
    "assert_full_coverage",
]


# =========================================================================== #
# Subclass Discovery (for completeness checks)                               #
# =========================================================================== #
def all_concrete_subclasses(base: type) -> set[type]:
    """Recursively collect every concrete (non-abstract) subclass of ``base``."""
    subclasses = set()
    for subclass in base.__subclasses__():
        if not getattr(subclass, "__abstractmethods__", None):
            subclasses.add(subclass)
        subclasses |= all_concrete_subclasses(subclass)
    return subclasses


def assert_full_coverage(
    base: type, tested: set[type], *, exclude: AbstractSet[type] = frozenset()
) -> None:
    """
    Assert that every concrete subclass of ``base`` has a corresponding test class.

    Parameters
    ----------
    base
        The abstract base class to sweep (e.g. ``Lightcurve``).
    tested
        The ``model_class`` values actually exercised by ``Test*`` classes in
        the calling module.
    exclude
        Concrete subclasses that are deliberately not covered (e.g.
        ``ComposedSpectralModel`` itself, which is an extension point rather
        than a real model).
    """
    missing = all_concrete_subclasses(base) - tested - exclude
    assert not missing, (
        f"The following concrete {base.__name__} subclasses have no "
        f"corresponding test class: {sorted(c.__qualname__ for c in missing)}. "
        "Add a `class Test<Name>(...Contract): model_class = <Name>` for each."
    )


# =========================================================================== #
# Shared Helpers                                                              #
# =========================================================================== #
def _arbitrary_valid_value(parameter: Parameter, rng):
    """A physical value for ``parameter`` guaranteed to lie within its prior's support."""
    latent = np.atleast_1d(parameter.sample_latent_variable(size=1, rng=rng))
    return parameter.transform_latent_to_physical(latent)[0]


def _integration_bounds(model_class, cgs_params: dict) -> tuple[float, float]:
    """
    Finite-or-infinite ``(low, high)`` cgs frequency bounds to integrate a model over.

    Prefers the model's own ``frequency_min``/``frequency_max`` parameters
    (the true, exact support of a power-law-like shape) when present, since
    integrating a discontinuous cutoff over artificially wide bounds is both
    unnecessary and numerically harder. Falls back to the class's
    :attr:`~m4opt.models.core._base.Spectrum._DOMAIN` (possibly infinite)
    otherwise.
    """
    if "frequency_min" in cgs_params and "frequency_max" in cgs_params:
        return float(cgs_params["frequency_min"]), float(cgs_params["frequency_max"])

    lo, hi = model_class._DOMAIN
    return float(to_cgs_value(lo)), float(to_cgs_value(hi))


#: Names of parameters that mark an interior derivative discontinuity in a
#: shape's frequency dependence (e.g. a broken power law's kink). Passed to
#: :func:`scipy.integrate.quad` as explicit break points -- without them,
#: adaptive quadrature can converge to a badly wrong answer right at the kink.
_BREAKPOINT_PARAMETER_NAMES = ("break_frequency",)


def _integrate_over_frequency(
    model_class, cgs_params: dict, lo: float, hi: float, eval_method: str = "eval_cgs"
) -> float:
    """
    Numerically integrate ``model_class.<eval_method>(nu, t?, **cgs_params) dnu``.

    Integrates in log-frequency space (substituting :math:`\\nu = e^u`,
    :math:`d\\nu = \\nu\\,du`) rather than directly in linear frequency space.
    A linear-space quadrature either misses a Planck function's peak entirely
    (when ``hi`` is infinite/astronomically large) or wastes its samples
    outside a power law's comparatively narrow support -- both of which
    :func:`scipy.integrate.quad` handles fine once reparametrized to log
    space, where every model here varies smoothly across a similar number of
    e-folds. Any parameter named in :data:`_BREAKPOINT_PARAMETER_NAMES` is
    passed through as an explicit break point, since QUADPACK can otherwise
    converge to a badly wrong answer right at a derivative kink (e.g.
    `BrokenPowerLawSpectrum`'s ``break_frequency``).
    """
    # A floor of 1e-10 Hz, not `np.finfo(float).tiny`: literally-subnormal
    # frequencies drive `h*nu/(k_B*T)` to underflow to exactly 0.0 for any
    # of these models' temperature/energy scales, which is a degenerate
    # input no physically meaningful grid would ever produce.
    log_lo = np.log(lo) if lo > 0 else np.log(1e-10)
    log_hi = np.log(hi) if np.isfinite(hi) else np.log(1e30)
    log_breakpoints = [
        np.log(cgs_params[name])
        for name in _BREAKPOINT_PARAMETER_NAMES
        if name in cgs_params and lo < cgs_params[name] < hi
    ]

    method = getattr(model_class, eval_method)

    def integrand(u):
        nu = np.exp(u)
        return float(method(nu, **cgs_params)) * nu

    integral, _ = quad(
        integrand, log_lo, log_hi, points=log_breakpoints or None, limit=400
    )
    return integral


def _integrate_over_frequency_at_time(
    model_class, cgs_params: dict, lo: float, hi: float, t: float, eval_method: str
) -> float:
    """Like :func:`_integrate_over_frequency`, for a method of the form ``f(nu, t, **params)``."""
    # A floor of 1e-10 Hz, not `np.finfo(float).tiny`: literally-subnormal
    # frequencies drive `h*nu/(k_B*T)` to underflow to exactly 0.0 for any
    # of these models' temperature/energy scales, which is a degenerate
    # input no physically meaningful grid would ever produce.
    log_lo = np.log(lo) if lo > 0 else np.log(1e-10)
    log_hi = np.log(hi) if np.isfinite(hi) else np.log(1e30)
    log_breakpoints = [
        np.log(cgs_params[name])
        for name in _BREAKPOINT_PARAMETER_NAMES
        if name in cgs_params and lo < cgs_params[name] < hi
    ]

    method = getattr(model_class, eval_method)

    def integrand(u):
        nu = np.exp(u)
        return float(method(nu, t, **cgs_params)) * nu

    integral, _ = quad(
        integrand, log_lo, log_hi, points=log_breakpoints or None, limit=400
    )
    return integral


# =========================================================================== #
# Shared Contract: Construction, Parameter Storage                           #
# =========================================================================== #
class ModelContract:
    """
    Generic instantiation and parameter-storage checks shared by every model kind.

    Subclasses must set :attr:`model_class` to a concrete
    :class:`~m4opt.models.core._base.Lightcurve`,
    :class:`~m4opt.models.core._base.Spectrum`, or
    :class:`~m4opt.models.core._base.SpectralModel` subclass.
    """

    #: The concrete model class under test. Must be overridden.
    model_class: ClassVar[type | None] = None

    #: Fixed RNG seed used throughout, for reproducible sampling.
    seed = 20240101

    # ----------------------------------- #
    # Construction                        #
    # ----------------------------------- #
    def test_instantiate_default(self):
        instance = self.model_class()
        assert isinstance(instance, self.model_class)

    def test_repr_lists_all_parameters(self):
        instance = self.model_class()
        text = repr(instance)
        assert instance.__class__.__name__ in text
        for name in self.model_class._DEFAULT_PARAMETERS:
            assert name in text

    # ----------------------------------- #
    # Mapping Interface                   #
    # ----------------------------------- #
    def test_mapping_interface(self):
        instance = self.model_class()
        names = set(self.model_class._DEFAULT_PARAMETERS)
        assert set(instance) == names
        assert len(instance) == len(names)
        for name in names:
            assert isinstance(instance[name], Parameter)

    def test_setitem_raises(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        name = next(iter(instance))
        with pytest.raises(TypeError):
            instance[name] = instance[name]

    def test_delitem_raises(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        name = next(iter(instance))
        with pytest.raises(TypeError):
            del instance[name]

    # ----------------------------------- #
    # Constructor Overrides                #
    # ----------------------------------- #
    def test_unknown_override_raises_keyerror(self):
        with pytest.raises(KeyError):
            self.model_class(_not_a_real_parameter_=1.0)

    def test_override_with_plain_value_fixes_parameter(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        name = next(iter(instance))
        value = _arbitrary_valid_value(instance[name], self.seed)

        fixed = self.model_class(**{name: value})
        assert fixed[name].is_fixed

    def test_override_with_parameter_replaces_it(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        name = next(iter(instance))
        replacement = copy.deepcopy(instance[name])
        replacement.fix(_arbitrary_valid_value(instance[name], self.seed))

        replaced = self.model_class(**{name: replacement})
        assert replaced[name] is replacement

    # ----------------------------------- #
    # Copying                             #
    # ----------------------------------- #
    def test_copy_shares_parameter_objects(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        clone = copy.copy(instance)
        name = next(iter(instance))
        assert clone[name] is instance[name]

    def test_deepcopy_creates_independent_parameters(self):
        instance = self.model_class()
        if len(instance) == 0:
            pytest.skip("model has no parameters")
        clone = copy.deepcopy(instance)
        name = next(iter(instance))
        assert clone[name] is not instance[name]

        clone[name].fix(_arbitrary_valid_value(clone[name], self.seed))
        assert not instance[name].is_fixed

    # ----------------------------------- #
    # Parameter Packing                   #
    # ----------------------------------- #
    def test_pack_unpack_roundtrip(self):
        cls = self.model_class
        values = {name: 1.0 for name in cls._DEFAULT_PARAMETERS}
        packed = cls.pack_params_to_arrays(**values)
        assert cls.unpack_params_from_arrays(*packed) == values

    def test_pack_missing_parameter_raises(self):
        cls = self.model_class
        if not cls._DEFAULT_PARAMETERS:
            pytest.skip("model has no parameters")
        with pytest.raises(KeyError):
            cls.pack_params_to_arrays()

    # ----------------------------------- #
    # Sampling                            #
    # ----------------------------------- #
    def test_sample_parameters_shape_and_finiteness(self):
        instance = self.model_class()
        size = 7
        samples = instance.sample_parameters(size=size, rng=self.seed)
        assert set(samples) == set(instance)
        for values in samples.values():
            arr = values.value if isinstance(values, Quantity) else np.asarray(values)
            assert np.shape(arr) == (size,)
            assert np.all(np.isfinite(np.asarray(arr, dtype=np.float64)))


# =========================================================================== #
# Lightcurve Contract                                                        #
# =========================================================================== #
class LightcurveContract(ModelContract):
    """Generic checks for :class:`~m4opt.models.core._base.Lightcurve` subclasses."""

    #: Time grid spanning every built-in lightcurve's characteristic
    #: rise/decay timescale (hours to ~months). Override per-subclass if a
    #: particular model needs a different range.
    t_grid: Quantity = np.geomspace(1e-2, 1e4, 64) * u.day

    def _sampled_params(self):
        instance = self.model_class()
        return instance.sample_parameters(size=1, rng=self.seed)

    def test_eval_returns_power_unit(self):
        params = self._sampled_params()
        result = self.model_class.eval(self.t_grid, **params)
        assert result.unit.is_equivalent(u.erg / u.s)

    def test_eval_finite_and_nonnegative(self):
        params = self._sampled_params()
        cgs_params = {name: to_cgs_value(value) for name, value in params.items()}
        result = self.model_class.eval_cgs(to_cgs_value(self.t_grid), **cgs_params)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_eval_log_cgs_matches_eval_cgs(self):
        params = self._sampled_params()
        cgs_params = {name: to_cgs_value(value) for name, value in params.items()}
        t_cgs = to_cgs_value(self.t_grid)
        np.testing.assert_allclose(
            np.exp(self.model_class.eval_log_cgs(t_cgs, **cgs_params)),
            self.model_class.eval_cgs(t_cgs, **cgs_params),
        )

    def test_eval_matches_eval_cgs(self):
        params = self._sampled_params()
        cgs_params = {name: to_cgs_value(value) for name, value in params.items()}
        t_cgs = to_cgs_value(self.t_grid)
        np.testing.assert_allclose(
            self.model_class.eval(self.t_grid, **params).cgs.value,
            self.model_class.eval_cgs(t_cgs, **cgs_params),
        )

    def test_eval_log_requires_time_quantity(self):
        params = self._sampled_params()
        with pytest.raises(TypeError):
            self.model_class.eval_log(1.0, **params)

    def test_eval_from_arrays_matches_eval(self):
        params = self._sampled_params()
        packed = self.model_class.pack_params_to_arrays(**params)
        np.testing.assert_allclose(
            self.model_class.eval_from_arrays(self.t_grid, *packed).cgs.value,
            self.model_class.eval(self.t_grid, **params).cgs.value,
        )

    def test_simulate_shape_and_finiteness(self):
        instance = self.model_class()
        t_scalar = self.t_grid[len(self.t_grid) // 2]
        result = instance.simulate(t_scalar, size=5, rng=self.seed)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result.cgs.value))


# =========================================================================== #
# Spectrum Contract                                                          #
# =========================================================================== #
class SpectrumContract(ModelContract):
    """Generic checks for :class:`~m4opt.models.core._base.Spectrum` subclasses."""

    #: Frequency grid spanning infrared through X-ray. Override per-subclass
    #: if a particular model needs a different range.
    nu_grid: Quantity = np.geomspace(1e10, 1e20, 64) * u.Hz

    def _sampled_params(self):
        instance = self.model_class()
        return instance.sample_parameters(size=1, rng=self.seed)

    def _scalar_cgs_params(self):
        params = self._sampled_params()
        return {name: to_cgs_value(value)[0] for name, value in params.items()}

    def test_eval_returns_shape_unit(self):
        params = self._sampled_params()
        result = self.model_class.eval(self.nu_grid, **params)
        assert result.unit.is_equivalent(u.Hz**-1)

    def test_eval_finite_and_nonnegative(self):
        cgs_params = self._scalar_cgs_params()
        result = self.model_class.eval_cgs(to_cgs_value(self.nu_grid), **cgs_params)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_eval_log_cgs_matches_eval_cgs(self):
        cgs_params = self._scalar_cgs_params()
        nu_cgs = to_cgs_value(self.nu_grid)
        np.testing.assert_allclose(
            np.exp(self.model_class.eval_log_cgs(nu_cgs, **cgs_params)),
            self.model_class.eval_cgs(nu_cgs, **cgs_params),
        )

    def test_eval_log_requires_frequency_quantity(self):
        params = self._sampled_params()
        with pytest.raises(TypeError):
            self.model_class.eval_log(1.0, **params)

    def test_normalization_matches_numeric_integral(self):
        cgs_params = self._scalar_cgs_params()
        lo, hi = _integration_bounds(self.model_class, cgs_params)

        expected = _integrate_over_frequency(self.model_class, cgs_params, lo, hi)
        actual = self.model_class.eval_normalization_cgs(**cgs_params)
        np.testing.assert_allclose(actual, expected, rtol=1e-3)

    def test_simulate_shape_and_finiteness(self):
        instance = self.model_class()
        nu_scalar = self.nu_grid[len(self.nu_grid) // 2]
        result = instance.simulate(nu_scalar, size=5, rng=self.seed)
        assert result.shape == (5,)
        assert np.all(np.isfinite(result.value))


# =========================================================================== #
# SpectralModel Contract                                                     #
# =========================================================================== #
class SpectralModelContract(ModelContract):
    """
    Generic checks for :class:`~m4opt.models.core._base.SpectralModel` subclasses.

    Covers both plain ``SpectralModel`` subclasses and
    :class:`~m4opt.models.core._base.ComposedSpectralModel` subclasses --
    both expose exactly the same public interface.
    """

    nu_grid: Quantity = np.geomspace(1e13, 1e17, 32) * u.Hz
    t_grid: Quantity = np.geomspace(1e-1, 2e2, 24) * u.day
    redshift = 0.05
    luminosity_distance: Quantity = 300 * u.Mpc

    def _sampled_params(self):
        instance = self.model_class()
        return instance.sample_parameters(size=1, rng=self.seed)

    def _scalar_cgs_params(self):
        params = self._sampled_params()
        return {name: to_cgs_value(value)[0] for name, value in params.items()}

    # ----------------------------------- #
    # L_nu(nu, t)                          #
    # ----------------------------------- #
    def test_eval_returns_spectral_luminosity_unit(self):
        params = self._sampled_params()
        result = self.model_class.eval(self.nu_grid[0], self.t_grid[0], **params)
        assert result.unit.is_equivalent(u.erg / u.s / u.Hz)

    def test_eval_finite_and_nonnegative(self):
        cgs_params = self._scalar_cgs_params()
        nu_cgs = to_cgs_value(self.nu_grid)[:, np.newaxis]
        t_cgs = to_cgs_value(self.t_grid)[np.newaxis, :]
        result = self.model_class.eval_cgs(nu_cgs, t_cgs, **cgs_params)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)

    def test_eval_log_cgs_matches_eval_cgs(self):
        cgs_params = self._scalar_cgs_params()
        nu_cgs = to_cgs_value(self.nu_grid)
        t_cgs = to_cgs_value(self.t_grid[0])
        np.testing.assert_allclose(
            np.exp(self.model_class.eval_log_cgs(nu_cgs, t_cgs, **cgs_params)),
            self.model_class.eval_cgs(nu_cgs, t_cgs, **cgs_params),
        )

    def test_eval_log_requires_quantities(self):
        params = self._sampled_params()
        with pytest.raises(TypeError):
            self.model_class.eval_log(1.0, self.t_grid[0], **params)
        with pytest.raises(TypeError):
            self.model_class.eval_log(self.nu_grid[0], 1.0, **params)

    def test_eval_from_arrays_matches_eval(self):
        params = self._sampled_params()
        packed = self.model_class.pack_params_to_arrays(**params)
        np.testing.assert_allclose(
            self.model_class.eval_from_arrays(
                self.nu_grid, self.t_grid[0], *packed
            ).cgs.value,
            self.model_class.eval(self.nu_grid, self.t_grid[0], **params).cgs.value,
        )

    # ----------------------------------- #
    # Bolometric / Normalized Shape       #
    # ----------------------------------- #
    def test_eval_spectrum_integrates_to_one(self):
        cgs_params = self._scalar_cgs_params()
        lo, hi = _integration_bounds(self.model_class, cgs_params)
        t0 = to_cgs_value(self.t_grid[len(self.t_grid) // 2])

        integral = _integrate_over_frequency_at_time(
            self.model_class, cgs_params, lo, hi, t0, "eval_spectrum_cgs"
        )
        np.testing.assert_allclose(integral, 1.0, rtol=1e-2)

    def test_eval_bolometric_matches_frequency_integral(self):
        cgs_params = self._scalar_cgs_params()
        lo, hi = _integration_bounds(self.model_class, cgs_params)
        t0 = to_cgs_value(self.t_grid[len(self.t_grid) // 2])

        expected = _integrate_over_frequency_at_time(
            self.model_class, cgs_params, lo, hi, t0, "eval_cgs"
        )
        actual = self.model_class.eval_bolometric_cgs(t0, **cgs_params)
        np.testing.assert_allclose(actual, expected, rtol=1e-2)

    # ----------------------------------- #
    # Synthetic Spectrum Generation       #
    # ----------------------------------- #
    def test_generate_spectrum_model_matches_eval(self):
        cgs_params = self._scalar_cgs_params()
        model = self.model_class.generate_spectrum_model(**cgs_params)
        nu0 = to_cgs_value(self.nu_grid[0])
        t0 = to_cgs_value(self.t_grid[0])
        np.testing.assert_allclose(
            model(nu0, t0),
            self.model_class.eval_cgs(nu0, t0, **cgs_params),
        )

    # ----------------------------------- #
    # Observed Flux / Magnitude           #
    # ----------------------------------- #
    def test_flux_matches_eval_at_zero_redshift(self):
        params = self._sampled_params()
        t0 = self.t_grid[0]
        flux = self.model_class.flux(
            self.nu_grid,
            t0,
            redshift=0.0,
            luminosity_distance=self.luminosity_distance,
            **params,
        )
        expected = self.model_class.eval(self.nu_grid, t0, **params) / (
            4 * np.pi * self.luminosity_distance**2
        )
        np.testing.assert_allclose(flux.cgs.value, expected.cgs.value, rtol=1e-6)

    def test_flux_log_requires_quantities(self):
        params = self._sampled_params()
        with pytest.raises(TypeError):
            self.model_class.flux_log(1.0, self.t_grid[0], redshift=0.0, **params)

    def test_mag_consistent_with_flux(self):
        params = self._sampled_params()
        t0 = self.t_grid[0]
        common = dict(
            redshift=self.redshift,
            luminosity_distance=self.luminosity_distance,
            **params,
        )
        flux = self.model_class.flux(self.nu_grid, t0, **common)
        mag = self.model_class.mag(self.nu_grid, t0, **common)
        expected = -2.5 * np.log10(flux.cgs.value / AB_MAG_ZERO_POINT)
        np.testing.assert_allclose(mag.value, expected)

    def test_flux_band_and_mag_band_are_finite(self):
        params = self._sampled_params()
        t0 = self.t_grid[0]
        throughput = np.ones_like(self.nu_grid.value)
        common = dict(
            redshift=self.redshift,
            luminosity_distance=self.luminosity_distance,
            **params,
        )
        flux_band = self.model_class.flux_band(self.nu_grid, throughput, t0, **common)
        mag_band = self.model_class.mag_band(self.nu_grid, throughput, t0, **common)
        assert np.isfinite(flux_band.cgs.value)
        assert np.isfinite(mag_band.value)

    # ----------------------------------- #
    # Simulation                          #
    # ----------------------------------- #
    def test_simulate_shape_and_finiteness(self):
        instance = self.model_class()
        result = instance.simulate(
            self.nu_grid[0], self.t_grid[0], size=5, rng=self.seed
        )
        assert result.shape == (5,)
        assert np.all(np.isfinite(result.cgs.value))
