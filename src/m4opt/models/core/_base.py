r"""
Base class for time- and frequency-dependent spectral models.

A :class:`SpectralModel` describes how a source's spectral luminosity,
:math:`L_\nu(\nu, t)`, evolves with observer-frame frequency :math:`\nu` and
time since explosion :math:`t`. From that single quantity, this base class
derives everything else a user typically needs to turn a physical model into
something observable: a bolometric light curve, a normalized spectral shape,
a redshifted and distance-diluted flux, a throughput-weighted band flux, AB
magnitudes, and a :class:`~synphot.SourceSpectrum` ready to feed into a
detector simulation.

To define a new model, subclass :class:`SpectralModel` and implement
:meth:`~SpectralModel._eval`, the natural-log spectral luminosity in cgs
units.

Every other quantity (fluxes, magnitudes, band-integrated fluxes, bolometric
luminosity) is derived automatically. All of these public methods come in a
consistent family of four, distinguished by suffix:

============ ================================= ===================
Suffix       Inputs / outputs                  Example
============ ================================= ===================
``_log_cgs`` unit-free numbers, natural log     :meth:`SpectralModel.eval_log_cgs`
``_log``     physical :class:`~astropy.units.Quantity`, natural log :meth:`SpectralModel.eval_log`
``_cgs``     unit-free numbers, linear scale    :meth:`SpectralModel.eval_cgs`
(none)       physical ``Quantity``, linear scale :meth:`SpectralModel.eval`
============ ================================= ===================

:class:`Lightcurve` and :class:`Spectrum` are the time-only and
frequency-only halves of the same idea: a :class:`Lightcurve` is just
:math:`L_\mathrm{bol}(t)`, and a :class:`Spectrum` is just a shape
:math:`S(\nu)` (not necessarily normalized to 1). :class:`ComposedSpectralModel`
combines one of each into a full :class:`SpectralModel`, with
:math:`L_\nu(\nu, t) = L_\mathrm{bol}(t) \cdot S(\nu) / \int S(\nu')\,d\nu'`.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from copy import copy, deepcopy
from typing import ClassVar, Self

import numpy as np
from astropy import units as u
from astropy.cosmology import FLRW
from astropy.modeling import Model
from astropy.units import Quantity
from scipy.integrate import quad_vec
from synphot import SourceSpectrum
from synphot import units as synphot_units

from .._cosmology import resolve_cosmological_distances
from .._typing import (
    CGSParameterValue,
    FloatArray,
    FloatResult,
    NumericalInput,
    OverrideValue,
    ParameterSamples,
    ParameterValue,
    PhysicalInput,
    RNGInput,
)
from .._utils import (
    _BOL_FLUX_UNIT,
    _BOL_LUM_UNIT,
    _SED_SHAPE_UNIT,
    _SPEC_FLUX_UNIT,
    _SPEC_LUM_UNIT,
    AB_MAG_ZERO_POINT,
    H_CGS,
    hz_per_unit,
    model_class_from_kernel,
    to_cgs_value,
)
from ._parameters import Parameter

__all__ = ["ComposedSpectralModel", "Lightcurve", "SpectralModel", "Spectrum"]


class _ModelBase(Mapping[str, Parameter], ABC):
    r"""
    Shared parameter storage, construction, and bookkeeping.

    Common base for :class:`SpectralModel`, :class:`Lightcurve`, and
    :class:`Spectrum`. Houses everything that doesn't depend on what a
    subclass physically computes: parameter declaration and validation
    (:attr:`_DEFAULT_PARAMETERS`, :attr:`_DOMAIN`), construction and
    copying, the :class:`~collections.abc.Mapping` interface, parameter
    packing, and parameter sampling. A subclass supplies only the physics --
    its own ``_eval``-style abstract method(s) mapping numerical inputs to a
    natural-log cgs result, the public ``eval*`` family that wraps them (see
    the module docstring for that four-way convention), and
    ``simulate``.

    Not part of the public API; use :class:`SpectralModel`,
    :class:`Lightcurve`, or :class:`Spectrum` instead.
    """

    # -------------------------------------- #
    # Class-Level Parameters                 #
    # -------------------------------------- #
    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {}
    """dict of Parameter: This model's parameters, and their default order.

    Deep-copied into each instance's own parameter set upon construction, so
    mutating one instance's parameters never affects the class default or any
    other instance.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity] | None] = None
    """tuple of Quantity, or None: The ``(low, high)`` frequency range a subclass integrates over, if any.

    ``None`` (the default) means this subclass has no frequency-domain
    integral to bound, and :meth:`__init_subclass__` skips the associated
    validation entirely -- true of :class:`Lightcurve`, which has no notion
    of frequency at all. :class:`SpectralModel` and :class:`Spectrum`
    override this with an actual ``(low, high)`` Quantity pair; see their
    own ``_DOMAIN`` documentation for what it bounds.
    """

    # -------------------------------------- #
    # Subclass Validation                    #
    # -------------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """Validate a subclass's :attr:`_DEFAULT_PARAMETERS` and, if set, :attr:`_DOMAIN`."""
        super().__init_subclass__(**kwargs)

        for name, parameter in cls._DEFAULT_PARAMETERS.items():
            if not isinstance(parameter, Parameter):
                raise TypeError(
                    f"{cls.__name__}._DEFAULT_PARAMETERS[{name!r}] must be a Parameter "
                    f"instance, got {type(parameter).__name__}."
                )

        if cls._DOMAIN is None:
            return

        low, high = cls._DOMAIN
        if not (isinstance(low, Quantity) and isinstance(high, Quantity)):
            raise TypeError(
                f"{cls.__name__}._DOMAIN must be a (low, high) pair of Quantity objects."
            )
        if (
            low.unit.physical_type != "frequency"
            or high.unit.physical_type != "frequency"
        ):
            raise TypeError(
                f"{cls.__name__}._DOMAIN must be expressed in frequency units."
            )
        if high <= low:
            raise ValueError(
                f"{cls.__name__}._DOMAIN must satisfy high > low, got ({low}, {high})."
            )

    # -------------------------------------- #
    # Construction and Copying               #
    # -------------------------------------- #
    def _init_parameters(self, overrides: Mapping[str, OverrideValue]) -> None:
        """Deep-copy :attr:`_DEFAULT_PARAMETERS` and apply constructor ``overrides`` on top."""
        self._parameters = deepcopy(self._DEFAULT_PARAMETERS)

        unknown = [name for name in overrides if name not in self._parameters]
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__} has no parameter(s) named {unknown}. "
                f"Valid parameters are {tuple(self._parameters)}."
            )

        for name, value in overrides.items():
            if isinstance(value, Parameter):
                self._parameters[name] = value
            else:
                self._parameters[name].fix(value)

    def __init__(self, **overrides: OverrideValue) -> None:
        """
        Create a model instance, optionally overriding some of its default parameters.

        Parameters
        ----------
        **overrides
            Per-parameter overrides, keyed by parameter name (must match one
            of this model's parameter names). Passing a plain value (a
            :class:`~astropy.units.Quantity`, ``float``, or ``int``) fixes
            that parameter to it; passing a
            :class:`~m4opt.models.core._parameters.Parameter` replaces the
            default parameter entirely (e.g. to use a different prior). In
            the latter case, the *same* ``Parameter`` instance is stored
            (not copied) -- passing one object to two model instances links
            them, so that fixing or sampling the parameter through either
            model affects both.

        Raises
        ------
        KeyError
            If ``overrides`` names a parameter this model doesn't have.
        """
        super().__init__()

        self._init_parameters(overrides)

    def __copy__(self) -> Self:
        new = self.__class__.__new__(self.__class__)
        new._parameters = copy(self._parameters)
        return new

    def __deepcopy__(self, memo: dict) -> Self:
        if id(self) in memo:
            return memo[id(self)]

        new = self.__class__.__new__(self.__class__)
        new._parameters = deepcopy(self._parameters, memo)
        memo[id(self)] = new
        return new

    # -------------------------------------- #
    # Mapping Interface                      #
    # -------------------------------------- #
    def __len__(self) -> int:
        return len(self._parameters)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def __setitem__(self, key: str, value: Parameter) -> None:
        raise TypeError(
            "Model parameters cannot be replaced directly. Modify the existing parameter instead."
        )

    def __delitem__(self, key: str) -> None:
        raise TypeError("Model parameters cannot be deleted.")

    def __repr__(self) -> str:
        rows = []
        for name, parameter in self._parameters.items():
            if parameter.is_fixed:
                rows.append(f"    {name}: fixed={parameter.fixed_value!r}")
            else:
                rows.append(f"    {name}: free, prior={parameter.prior.name}")

        return f"{self.__class__.__name__}(\n" + "\n".join(rows) + "\n)"

    # -------------------------------------- #
    # Parameter Packing                       #
    # -------------------------------------- #
    @classmethod
    def pack_params_to_arrays(
        cls, **parameters: ParameterValue
    ) -> tuple[ParameterValue, ...]:
        """
        Convert a dict of parameter values into an ordered sequence.

        The output is a plain tuple of the input values, reordered to match
        this model's parameter order -- not stacked into a single array
        (parameters may carry different, incompatible units).

        Parameters
        ----------
        **parameters
            Parameter values, keyed by name. Must supply a value for every
            one of this model's parameters.

        Returns
        -------
        tuple
            The values from ``parameters``, in this model's parameter order.

        Raises
        ------
        KeyError
            If ``parameters`` is missing a value for one or more of this
            model's parameters.

        See Also
        --------
        unpack_params_from_arrays : The inverse conversion.
        """
        missing = [name for name in cls._DEFAULT_PARAMETERS if name not in parameters]
        if missing:
            raise KeyError(
                f"Missing value(s) for parameter(s) {missing} required by {cls.__name__}."
            )

        return tuple(parameters[name] for name in cls._DEFAULT_PARAMETERS)

    @classmethod
    def unpack_params_from_arrays(
        cls, *parameters: ParameterValue
    ) -> dict[str, ParameterValue]:
        """
        Convert an ordered sequence of parameter values back into a dict.

        The inverse of :meth:`pack_params_to_arrays`.

        Parameters
        ----------
        *parameters
            Parameter values, one per parameter of this model, in this
            model's parameter order.

        Returns
        -------
        dict
            ``{name: value}`` for each of this model's parameters.

        Raises
        ------
        ValueError
            If the number of positional values doesn't match the number of
            parameters.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion.
        """
        names = tuple(cls._DEFAULT_PARAMETERS)
        if len(parameters) != len(names):
            raise ValueError(
                f"{cls.__name__} expected {len(names)} positional parameter "
                f"value(s) {names}, got {len(parameters)}."
            )

        return dict(zip(names, parameters))

    # -------------------------------------- #
    # Parameter Sampling                     #
    # -------------------------------------- #
    def sample_parameters(
        self,
        size: int = 1,
        *,
        rng: RNGInput = None,
        parameters: list[str] | None = None,
    ) -> ParameterSamples:
        """
        Draw random samples of some or all of this model's parameters.

        Parameters
        ----------
        size
            Number of samples to draw per parameter.
        rng
            Random-number source, forwarded to each
            :meth:`~m4opt.models.core._parameters.Parameter.sample`. Passing
            a shared :class:`~numpy.random.Generator` is recommended so that
            every parameter's draws come from the same reproducible stream.
        parameters
            Names of the parameters to sample. If ``None`` (the default),
            every parameter is sampled.

        Returns
        -------
        dict
            ``{name: samples}`` for each requested parameter, each with
            shape ``(size,)``.
        """
        if parameters is None:
            parameters = list(self._parameters.keys())

        return {
            parameter_name: parameter.sample(size=size, rng=rng)
            for parameter_name, parameter in self._parameters.items()
            if parameter_name in parameters
        }


class Lightcurve(_ModelBase):
    r"""
    Abstract base class for time-dependent bolometric luminosity models.

    A :class:`Lightcurve` describes how a source's bolometric
    (frequency-integrated) luminosity, :math:`L_\mathrm{bol}(t)`, evolves
    with time since explosion :math:`t`. It shares :class:`SpectralModel`'s
    parameter storage, evaluation-family conventions, and sampling
    machinery, but carries none of :class:`SpectralModel`'s :math:`\nu`-dependent or
    observed-frame machinery (spectral shape, flux, band flux, magnitudes).

    To define a new model, subclass :class:`Lightcurve` and implement
    :meth:`_eval`, the natural log of :math:`L_\mathrm{bol}(t)` in cgs
    units. Every other quantity (the ``*_log_cgs``/``*_log``/``*_cgs``/plain
    family described in the module docstring) is derived automatically.

    Pairing a :class:`Lightcurve` with a :class:`Spectrum` via
    :class:`ComposedSpectralModel` turns it into a full :math:`L_\nu(\nu,
    t)` :class:`SpectralModel`.

    See Also
    --------
    Spectrum : The frequency-only counterpart this mirrors.
    ComposedSpectralModel : Combines a Lightcurve and a Spectrum into a full SED.
    """

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(cls, t: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`L_\mathrm{bol}(t)`, in cgs units.

        This is the one method every model must implement. ``t`` and every
        parameter value are combined using plain NumPy broadcasting -- no
        axes are inserted automatically.

        Parameters
        ----------
        t
            Time since explosion, in seconds. Always non-negative.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        ...

    @classmethod
    def eval_log_cgs(
        cls, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        r"""
        Natural log of the bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s. A scalar
            is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval(t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(cls, t: Quantity, **parameters: ParameterValue) -> FloatResult:
        r"""
        Natural log of the bolometric luminosity, given physical-unit inputs.

        Parameters
        ----------
        t
            Time since explosion, with time units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.

        Raises
        ------
        TypeError
            If ``t`` is not a Quantity with time units.
        """
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_log_cgs(t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(
        cls, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """Bolometric luminosity, taking and returning plain cgs numbers. See :meth:`eval_log_cgs`."""
        return np.exp(cls.eval_log_cgs(t, **parameters))

    @classmethod
    def eval(cls, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the bolometric luminosity at the given time.

        Parameters
        ----------
        t
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        return np.exp(cls.eval_log(t, **parameters)) * _BOL_LUM_UNIT

    @classmethod
    def eval_from_arrays(cls, t: Quantity, *parameters: ParameterValue) -> Quantity:
        """
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(t, **cls.unpack_params_from_arrays(*parameters))``.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(t, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(self, t: Quantity, size: int = 1, *, rng: RNGInput = None) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given time.

        Equivalent to ``self.eval(t, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        t
            Time since explosion. See :meth:`eval`.
        size
            Number of realizations to draw.
        rng
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(t, **self.sample_parameters(size=size, rng=rng))


class Spectrum(_ModelBase):
    r"""
    Abstract base class for frequency-dependent spectral shape models.

    A :class:`Spectrum` describes how a source's light is distributed across
    frequency, as a *shape* :math:`S(\nu)`. It
    shares :class:`SpectralModel`'s parameter storage, evaluation-family
    conventions, and sampling machinery, but -- having no notion of time,
    redshift, or distance -- carries none of :class:`SpectralModel`'s
    :math:`t`-dependent or observed-frame machinery (bolometric luminosity,
    flux, band flux, magnitudes).

    To define a new model, subclass :class:`Spectrum` and implement
    :meth:`_eval`, the natural log of :math:`S(\nu)` in cgs units. Unlike
    :class:`SpectralModel`'s spectral shape, :math:`S(\nu)` need not
    integrate to any particular value by construction --
    :meth:`eval_normalization` computes whatever :math:`\int S(\nu)\,d\nu`
    actually is, which is exactly the factor :class:`ComposedSpectralModel`
    divides out to combine a :class:`Spectrum` with a :class:`Lightcurve`'s
    :math:`L_\mathrm{bol}(t)` into an exactly normalized :math:`L_\nu(\nu,
    t)`.

    See Also
    --------
    Lightcurve : The time-only counterpart this mirrors.
    ComposedSpectralModel : Combines a Lightcurve and a Spectrum into a full SED.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity]] = (0 * u.Hz, np.inf * u.Hz)
    """tuple of Quantity: The ``(low, high)`` frequency range integrated over
    when computing :meth:`eval_normalization` (see :meth:`_eval_normalization`).

    Only consulted by the default, numerical-quadrature implementation of
    :meth:`_eval_normalization`. A subclass that overrides that method with a
    closed-form expression does not need this to be meaningful.
    """

    # -------------------------------------- #
    # Spectral Shape: S(nu)                   #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(cls, nu: FloatArray, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`S(\nu)`, in cgs units.

        This is the one method every model must implement. ``nu`` and every
        parameter value are combined using plain NumPy broadcasting -- no
        axes are inserted automatically. There is no requirement that
        :math:`S(\nu)` integrate to any particular value; see
        :meth:`eval_normalization`.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`S(\nu)`, in 1/Hz.
        """
        ...

    @classmethod
    def eval_log_cgs(
        cls, nu: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        r"""
        Natural log of the spectral shape, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`S(\nu)`, in 1/Hz. A scalar is returned
            if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval(nu_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(cls, nu: Quantity, **parameters: ParameterValue) -> FloatResult:
        r"""
        Natural log of the spectral shape, given physical-unit inputs.

        Parameters
        ----------
        nu
            Frequency, with frequency units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`S(\nu)`, in 1/Hz.

        Raises
        ------
        TypeError
            If ``nu`` is not a Quantity with frequency units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_log_cgs(nu.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(
        cls, nu: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """Spectral shape, taking and returning plain cgs numbers. See :meth:`eval_log_cgs`."""
        return np.exp(cls.eval_log_cgs(nu, **parameters))

    @classmethod
    def eval(cls, nu: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the spectral shape at the given frequency.

        Parameters
        ----------
        nu
            Frequency at which to evaluate the shape.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu)`, in 1/Hz.
        """
        return np.exp(cls.eval_log(nu, **parameters)) * _SED_SHAPE_UNIT

    @classmethod
    def eval_from_arrays(cls, nu: Quantity, *parameters: ParameterValue) -> Quantity:
        """
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(nu, **cls.unpack_params_from_arrays(*parameters))``.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(nu, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Normalization: integral of S(nu) dnu    #
    # -------------------------------------- #
    @classmethod
    def _eval_normalization(cls, **parameters: CGSParameterValue) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`\int S(\nu)\,d\nu`, in cgs units.

        Default implementation: broadcasts ``parameters`` together, then
        numerically integrates :math:`\int \exp(\mathtt{\_eval}(\nu))\,d\nu`
        over :attr:`_DOMAIN` (:func:`scipy.integrate.quad_vec`, which
        evaluates the whole broadcast array at each trial frequency at once
        rather than looping over realizations one at a time).

        A model whose shape integral has a closed form (e.g. one already
        normalized to 1 by construction) should override this method
        directly, for both speed and exactness.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against one another.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`\int S(\nu)\,d\nu`, dimensionless.
        """
        param_arrays = [
            np.asarray(value, dtype=np.float64) for value in parameters.values()
        ]
        param_grids = (
            dict(zip(parameters, np.broadcast_arrays(*param_arrays)))
            if param_arrays
            else {}
        )

        lo, hi = cls._DOMAIN

        def integrand(nu: float) -> FloatArray:
            return np.exp(cls._eval(np.asarray(nu, dtype=np.float64), **param_grids))

        integral, _ = quad_vec(
            integrand, float(to_cgs_value(lo)), float(to_cgs_value(hi))
        )

        return np.log(integral)

    @classmethod
    def eval_normalization_log_cgs(cls, **parameters: CGSParameterValue) -> FloatResult:
        r"""
        Natural log of the shape's frequency integral, taking and returning plain cgs numbers.

        Parameters
        ----------
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`\int S(\nu)\,d\nu`. A scalar is
            returned if the result is 0-dimensional.
        """
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval_normalization(**cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_normalization_log(cls, **parameters: ParameterValue) -> FloatResult:
        """Natural log of the shape's frequency integral, given physical-unit inputs. See :meth:`eval_log`."""
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_normalization_log_cgs(**cgs_parameters)

    @classmethod
    def eval_normalization_cgs(cls, **parameters: CGSParameterValue) -> FloatResult:
        """The shape's frequency integral, taking and returning plain cgs numbers. See :meth:`eval_normalization_log_cgs`."""
        return np.exp(cls.eval_normalization_log_cgs(**parameters))

    @classmethod
    def eval_normalization(cls, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate :math:`\int S(\nu)\,d\nu`.

        This is exactly the factor by which :math:`S(\nu)` must be divided
        to turn it into a shape that integrates to 1 -- the normalization
        :class:`ComposedSpectralModel` applies when combining a
        :class:`Spectrum` with a :class:`Lightcurve`'s
        :math:`L_\mathrm{bol}(t)` to get :math:`L_\nu(\nu, t)`.

        Parameters
        ----------
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`\int S(\nu)\,d\nu`, dimensionless.
        """
        return (
            np.exp(cls.eval_normalization_log(**parameters)) * u.dimensionless_unscaled
        )

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(
        self, nu: Quantity, size: int = 1, *, rng: RNGInput = None
    ) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given frequency.

        Equivalent to ``self.eval(nu, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        nu
            Frequency at which to evaluate the model. See :meth:`eval`.
        size
            Number of realizations to draw.
        rng
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu)`, in 1/Hz.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(nu, **self.sample_parameters(size=size, rng=rng))


class SpectralModel(_ModelBase):
    r"""
    Abstract base class for time-dependent spectral energy distribution models.

    A ``SpectralModel`` represents the intrinsic spectral luminosity

    .. math::

        L_\nu(\nu, t),

    as a function of source-frame frequency :math:`\nu` and time since
    explosion :math:`t`. From this fundamental quantity, the base class
    provides a consistent interface for computing bolometric luminosities,
    normalized spectral shapes, observed flux densities, band-averaged
    fluxes, apparent magnitudes, and synthetic
    :class:`~synphot.SourceSpectrum` objects.


    Notes
    -----
    The core subclassing interface consists of:

    - :attr:`_DEFAULT_PARAMETERS` for declaring model parameters;
    - :attr:`_DOMAIN` for the frequency range used by the default bolometric
      integration; and
    - :meth:`_eval` for evaluating
      :math:`\log L_\nu(\nu, t)` in unit-stripped cgs coordinates.

    Every other public method -- including :meth:`as_astropy_model`
    and :meth:`as_source_spectrum`, which build synthetic spectra -- is
    derived automatically from :meth:`_eval` and requires no per-subclass
    override.

    Most user-facing quantities are available in parallel interfaces:

    ``*_log_cgs``
        Unit-stripped cgs inputs and logarithmic numerical output.

    ``*_log``
        Physical-unit inputs and logarithmic numerical output.

    ``*_cgs``
        Unit-stripped cgs inputs and linear numerical output.

    no suffix
        Physical-unit inputs and physical-unit output.

    Because magnitudes are already logarithmic quantities, their API consists
    only of unit-stripped ``*_cgs`` and unit-aware forms.

    All of the above are :class:`classmethod`\ s: they operate purely on the
    ``**parameters`` values passed in, not on any particular instance's
    stored parameter configuration. Only parameter *storage* -- inherited
    from :class:`_ModelBase` -- requires an instance.
    """

    _DOMAIN: ClassVar[tuple[Quantity, Quantity]] = (0 * u.Hz, np.inf * u.Hz)
    """tuple of Quantity: The ``(low, high)`` frequency range integrated over
    when computing the bolometric luminosity (see :meth:`_eval_bolometric`).

    Only consulted by the default, numerical-quadrature implementation of
    :meth:`_eval_bolometric`. A subclass that overrides that method with a
    closed-form expression does not need this to be meaningful.
    """

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    @abstractmethod
    def _eval(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        Evaluate the natural log of :math:`L_\nu(\nu, t)`, in cgs units.

        This is the one method every model must implement. ``nu``, ``t``,
        and every parameter value are combined using plain NumPy
        broadcasting -- no axes are inserted automatically. See the module
        docstring for what that means in practice.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        t
            Time since explosion, in seconds. Always non-negative.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu`` and ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        ...

    @classmethod
    def eval_log_cgs(
        cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """
        Natural log of the spectral luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        t
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the spectral luminosity, in erg/s/Hz. A
            scalar is returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval(nu_arr, t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_log(
        cls, nu: Quantity, t: Quantity, **parameters: ParameterValue
    ) -> FloatResult:
        """
        Natural log of the spectral luminosity, given physical-unit inputs.

        Parameters
        ----------
        nu
            Frequency, with frequency units.
        t
            Time since explosion, with time units.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the spectral luminosity, in erg/s/Hz.

        Raises
        ------
        TypeError
            If ``nu``/``t`` are not Quantities with frequency/time units.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_log_cgs(nu.cgs.value, t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_cgs(
        cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """Spectral luminosity, taking and returning plain cgs numbers. See :meth:`eval_log_cgs`."""
        return np.exp(cls.eval_log_cgs(nu, t, **parameters))

    @classmethod
    def eval(cls, nu: Quantity, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the spectral luminosity at the given frequency and time.

        Parameters
        ----------
        nu
            Frequency at which to evaluate the model.
        t
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\nu(\nu, t)`, in erg/s/Hz.
        """
        return np.exp(cls.eval_log(nu, t, **parameters)) * _SPEC_LUM_UNIT

    @classmethod
    def eval_from_arrays(
        cls, nu: Quantity, t: Quantity, *parameters: ParameterValue
    ) -> Quantity:
        """
        Positional-argument form of :meth:`eval`.

        Equivalent to ``cls.eval(nu, t, **cls.unpack_params_from_arrays(*parameters))``.
        Useful when parameter values are already stored as a plain sequence
        (e.g. rows of an array) rather than a dict.

        See Also
        --------
        pack_params_to_arrays : The inverse conversion, dict -> ordered sequence.
        unpack_params_from_arrays : Ordered sequence -> dict, used internally here.
        """
        return cls.eval(nu, t, **cls.unpack_params_from_arrays(*parameters))

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(
        cls, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the bolometric luminosity, in cgs units.

        Default implementation: broadcasts ``t`` and ``parameters`` together,
        then numerically integrates :math:`\int \exp(\mathtt{\_eval}(\nu,
        t))\,d\nu` over :attr:`_DOMAIN` (:func:`scipy.integrate.quad_vec`,
        which evaluates the whole broadcast array at each trial frequency at
        once rather than looping over realizations/times one at a time).

        A model for which this integral has a closed form should override
        this method directly, for both speed and exactness.

        Parameters
        ----------
        t
            Time since explosion, in seconds, broadcastable against
            ``parameters``.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        t_grid, *param_arrays = np.broadcast_arrays(
            np.asarray(t, dtype=np.float64), *parameters.values()
        )
        param_grids = dict(zip(parameters, param_arrays))

        lo, hi = cls._DOMAIN

        def integrand(nu: float) -> FloatArray:
            return np.exp(
                cls._eval(np.asarray(nu, dtype=np.float64), t_grid, **param_grids)
            )

        integral, _ = quad_vec(
            integrand, float(to_cgs_value(lo)), float(to_cgs_value(hi))
        )

        return np.log(integral)

    @classmethod
    def eval_bolometric_log_cgs(
        cls, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """
        Natural log of the bolometric luminosity, taking and returning plain cgs numbers.

        Parameters
        ----------
        t
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the bolometric luminosity, in erg/s. A scalar
            is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval_bolometric(t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_bolometric_log(
        cls, t: Quantity, **parameters: ParameterValue
    ) -> FloatResult:
        """Natural log of the bolometric luminosity, given physical-unit inputs. See :meth:`eval_log`."""
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_bolometric_log_cgs(t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_bolometric_cgs(
        cls, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """Bolometric luminosity, taking and returning plain cgs numbers. See :meth:`eval_log_cgs`."""
        return np.exp(cls.eval_bolometric_log_cgs(t, **parameters))

    @classmethod
    def eval_bolometric(cls, t: Quantity, **parameters: ParameterValue) -> Quantity:
        r"""
        Evaluate the bolometric luminosity at the given time.

        Parameters
        ----------
        t
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\mathrm{bol}(t)`, in erg/s.
        """
        return np.exp(cls.eval_bolometric_log(t, **parameters)) * _BOL_LUM_UNIT

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the normalized spectral shape, in cgs units.

        Default implementation: :math:`\log S(\nu, t) = \mathtt{\_eval}(\nu,
        t) - \mathtt{\_eval\_bolometric}(t)`, i.e. :math:`S(\nu, t) =
        L_\nu(\nu, t)/L_\mathrm{bol}(t)`, computed as a log-space subtraction
        rather than a linear-space division so it stays accurate regardless
        of :math:`L_\nu`'s dynamic range. This integrates to exactly 1 over
        :math:`\nu`, for any :math:`t`, by construction.

        A model whose spectral shape is already known independently of its
        bolometric integral (e.g. one built from a separately normalized
        template spectrum) should override this method directly, skipping
        the bolometric-integral subtraction entirely.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        t
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``/``t``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`S(\nu, t)`, in 1/Hz.
        """
        return cls._eval(nu, t, **parameters) - cls._eval_bolometric(t, **parameters)

    @classmethod
    def eval_spectrum_log_cgs(
        cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """
        Natural log of the normalized spectral shape, taking and returning plain cgs numbers.

        Parameters
        ----------
        nu
            Frequency, in Hz.
        t
            Time since explosion, in seconds.
        **parameters
            This model's parameter values, in cgs units. To evaluate several
            parameter realizations at once, give each parameter a leading
            batch axis.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the normalized spectral shape. A scalar is
            returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval_spectrum(nu_arr, t_arr, **cgs_parameters)
        return result.item() if result.ndim == 0 else result

    @classmethod
    def eval_spectrum_log(
        cls, nu: Quantity, t: Quantity, **parameters: ParameterValue
    ) -> FloatResult:
        """Natural log of the normalized spectral shape, given physical-unit inputs. See :meth:`eval_log`."""
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.eval_spectrum_log_cgs(nu.cgs.value, t.cgs.value, **cgs_parameters)

    @classmethod
    def eval_spectrum_cgs(
        cls, nu: NumericalInput, t: NumericalInput, **parameters: CGSParameterValue
    ) -> FloatResult:
        """Normalized spectral shape, taking and returning plain cgs numbers. See :meth:`eval_log_cgs`."""
        return np.exp(cls.eval_spectrum_log_cgs(nu, t, **parameters))

    @classmethod
    def eval_spectrum(
        cls, nu: Quantity, t: Quantity, **parameters: ParameterValue
    ) -> Quantity:
        r"""
        Evaluate the normalized spectral shape at the given frequency and time.

        Parameters
        ----------
        nu
            Frequency at which to evaluate the shape.
        t
            Time since explosion.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`S(\nu, t)`, in 1/Hz. Integrates to 1 over :math:`\nu`
            for any fixed :math:`t`.
        """
        return np.exp(cls.eval_spectrum_log(nu, t, **parameters)) * _SED_SHAPE_UNIT

    # -------------------------------------- #
    # Synthetic Spectrum Generation          #
    # -------------------------------------- #
    # At the m4opt.synphot level, operations are performed on Synphot / Astropy Model objects,
    # which are not immediately compatible with the machinery of the SpectralModel class. These
    # methods allow a user to provide a set of parameters and generate spectra objects.
    @classmethod
    def as_astropy_model(
        cls,
        x_type: str = "lambda",
        y_type: str = "lambda",
        *,
        y_kind: str = "energy",
        wave_unit: str | u.UnitBase = u.AA,
        freq_unit: str | u.UnitBase = u.Hz,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> Model:
        r"""
        Build an :class:`~astropy.modeling.Model` of this :class:`SpectralModel` for a given parameter set.

        This method is the single entry point for converting a :class:`SpectralModel` into
        a model compatible with :mod:`synphot` / :mod:`astropy.modeling` / :mod:`m4opt.synphot`.
        Several parameters can be modified to specify exactly what type of spectral model to
        generate:

        - ``x_type``: May be either ``"lambda"`` or ``"nu"`` to control what input the
          model expects.
        - ``y_type``: May be either ``"lambda"`` or ``"nu"``. If ``"lambda"``, then :math:`F_\lambda` is
          generated, otherwise :math:`F_\nu` is generated.
        - ``y_kind``: May be either ``"energy"`` or ``"photon"``. If ``"energy"``, then the output is
          an energy-flux density (:math:`F_\nu`/:math:`F_\lambda`). If ``"photon"``, then the output is
          a photon count flux density.

        Additionally, any of a number of cosmological parameters may be specified to provide the distance
        from the source. If any of these are provided, the model will generate a flux density. Otherwise
        a luminosity density is produced.

        Parameters
        ----------
        x_type
            ``"lambda"`` (the default) if the model's first input is a
            wavelength, or ``"nu"`` if it is a frequency.
        y_type
            ``"lambda"`` (the default) if the output is expressed per unit
            wavelength, or ``"nu"`` if per unit frequency.
        y_kind
            ``"energy"`` (the default) for an energy-flux density, or
            ``"photon"`` for a photon-count-flux density (dividing by the
            photon energy :math:`h\nu`).
        wave_unit
            The wavelength unit used wherever ``x_type``/``y_type`` is
            ``"lambda"``.
        freq_unit
            The frequency unit used wherever ``x_type``/``y_type`` is
            ``"nu"``.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            If any of ``redshift``/``luminosity_distance``/
            ``angular_diameter_distance``/``proper_distance`` is given,
            exactly one of them must be given; the rest (and
            ``cosmology``) are as in :meth:`flux_log`, and the output is
            the observed, diluted flux. If none of the four is given, the
            output is the rest-frame luminosity and ``cosmology`` is
            ignored.
        **parameters
            This model's parameter values, either
            :class:`~astropy.units.Quantity` or already unit-stripped cgs
            values (see :meth:`eval_log_cgs`). May carry leading batch
            axes.

        Returns
        -------
        ~astropy.modeling.Model
            Callable as ``model(x, t)``, with ``x``/``t`` either bare
            numbers (``x`` in ``wave_unit``/``freq_unit``, ``t`` in
            seconds) or :class:`~astropy.units.Quantity`. Unit-attached
            only when at least one of ``x``/``t`` was itself a
            ``Quantity`` (a property of
            :meth:`astropy.modeling.Model.__call__`, not something this
            method controls).

        Raises
        ------
        ValueError
            If ``x_type``/``y_type`` is not ``"lambda"``/``"nu"``, if
            ``y_kind`` is not ``"energy"``/``"photon"``, or if
            ``wave_unit``/``freq_unit`` is not a wavelength/frequency unit.
        """
        # Resolve and validate the x_type, y_type and y_kind parameters.
        if x_type not in ("nu", "lambda"):
            raise ValueError(f"x_type must be 'nu' or 'lambda', got {x_type!r}.")
        if y_type not in ("nu", "lambda"):
            raise ValueError(f"y_type must be 'nu' or 'lambda', got {y_type!r}.")
        if y_kind not in ("energy", "photon"):
            raise ValueError(f"y_kind must be 'energy' or 'photon', got {y_kind!r}.")

        x_is_wavelength = x_type == "lambda"
        y_is_wavelength = y_type == "lambda"

        x_unit = u.Unit(wave_unit if x_is_wavelength else freq_unit)
        y_unit = u.Unit(wave_unit if y_is_wavelength else freq_unit)

        x_coefficient = hz_per_unit(x_unit, is_wavelength=x_is_wavelength)
        y_coefficient = hz_per_unit(y_unit, is_wavelength=y_is_wavelength)

        # Determine if we are generating a flux density or luminosity. This is determined
        # by the specification / lack of the redshift or other cosmological parameters.
        _is_luminosity = (
            redshift is None
            and luminosity_distance is None
            and angular_diameter_distance is None
            and proper_distance is None
        )

        # Convert ALL of the provided parameters to their CGS value so that we
        # do not need to do on-the-fly unit conversions in performance critical
        # call sequences.
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        # Determine the "native" evaluation function. This is the F_nu(nu) in units of
        # erg / cm^2 / Hz / s if we are computing a flux, or a luminosity L_nu(nu) in
        # erg / Hz / s if we are computing a luminosity.
        #
        # Once this has been generated, we can modify it to generate the correct units
        # and other properties.
        if not _is_luminosity:
            # Resolve the cosmological distances and extract them.
            distances = resolve_cosmological_distances(
                redshift=redshift,
                luminosity_distance=luminosity_distance,
                angular_diameter_distance=angular_diameter_distance,
                proper_distance=proper_distance,
                cosmology=cosmology,
            )
            redshift_cgs = np.asarray(distances["redshift"], dtype=np.float64)
            luminosity_distance_cgs = distances["luminosity_distance"].cgs.value
            _denominator_unit = u.s**-1 * u.cm**-2

            # Construct the native evaluation function.
            def _eval_native(nu_hz: FloatArray, t: FloatArray) -> FloatResult:
                return cls.flux_cgs(
                    nu_hz, t, redshift_cgs, luminosity_distance_cgs, **cgs_parameters
                )
        else:
            # We are not producing a flux. We can just generate the function.
            _denominator_unit = u.s**-1

            def _eval_native(nu_hz: FloatArray, t: FloatArray) -> FloatResult:
                return cls.eval_cgs(nu_hz, t, **cgs_parameters)

        # Generate the modifications to the function to coerce x to nu.
        if x_is_wavelength:

            def _x_to_nu(x: FloatArray) -> FloatArray:
                return x_coefficient / x
        else:

            def _x_to_nu(x: FloatArray) -> FloatArray:
                return x_coefficient * x

        # Generate the modification to the function to coerce F_nu to y.
        if y_is_wavelength:

            def _to_dlambda(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y * (nu_hz**2 / y_coefficient))
        else:

            def _to_dlambda(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y * y_coefficient)

        if y_kind == "energy":
            _numerator_unit = u.erg

            def _to_output_flux(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return np.asarray(y)
        else:
            _numerator_unit = u.photon

            def _to_output_flux(nu_hz: FloatArray, y: FloatResult) -> FloatArray:
                return y / (H_CGS * nu_hz)

        # Generate the final evaluator.
        def _evaluate(x: FloatArray, t: FloatArray) -> FloatResult:
            nu_hz = _x_to_nu(x)
            y = _eval_native(nu_hz, t)
            y = _to_dlambda(nu_hz, y)
            y = _to_output_flux(nu_hz, y)
            return y

        output_unit = _numerator_unit / y_unit * _denominator_unit

        model_class = model_class_from_kernel(
            "_SpectralAstropyModel",
            inputs={("wave" if x_is_wavelength else "nu"): x_unit, "t": u.s},
            outputs={"y": output_unit},
            evaluate=_evaluate,
        )
        return model_class()

    @classmethod
    def as_source_spectrum(
        cls,
        t: PhysicalInput,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> SourceSpectrum:
        r"""
        Build a :class:`~synphot.SourceSpectrum` giving the observed flux at one fixed time :math:`t`.

        A thin wrapper around :meth:`as_astropy_model`: fixes ``t`` (so the
        result is a function of wavelength alone, the shape :mod:`synphot`
        requires), and fixes ``x_type``/``y_type``/``y_kind`` to
        wavelength-in/photon-out, in Angstrom -- because
        :class:`~synphot.SourceSpectrum` always samples its wrapped model
        in *wavelength* space, and always treats the model's raw return
        value as already being expressed in :mod:`synphot`'s internal
        :data:`~synphot.units.PHOTLAM` (*photon*-count flux density per
        unit wavelength, not the energy-flux :math:`F_\lambda` one might
        expect) -- :meth:`~synphot.SourceSpectrum.__call__` does not
        consult a wrapped model's declared output units to convert.

        Requires an observed (diluted) flux -- at least one of
        ``redshift``/the distance keywords -- since a
        :class:`~synphot.SourceSpectrum` is meant to be a real per-area
        flux for :class:`~m4opt.synphot.Detector` to consume, not a
        rest-frame luminosity. This does not apply any foreground
        attenuation (e.g. Milky Way dust) -- multiply the returned
        :class:`~synphot.SourceSpectrum` by a separate extinction spectral
        element for that, as :class:`~m4opt.synphot.extinction.DustExtinction`
        does.

        Parameters
        ----------
        t
            Observed time since explosion, either a
            :class:`~astropy.units.Quantity` with time units or an already
            unit-stripped cgs (seconds) value. May carry leading batch
            axes, broadcastable against ``parameters``.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            Exactly one of ``redshift`` or the three distance keywords must
            be given; the rest are derived from it using ``cosmology``. See
            :meth:`as_astropy_model`.
        **parameters
            This model's parameter values, either
            :class:`~astropy.units.Quantity` or already unit-stripped cgs
            values (see :meth:`eval_log_cgs`). May carry leading batch
            axes.

        Returns
        -------
        ~synphot.SourceSpectrum
            Callable as ``spectrum(wave)``, returning the observed flux
            density at ``t``.

        Raises
        ------
        ValueError
            If none of ``redshift``/``luminosity_distance``/
            ``angular_diameter_distance``/``proper_distance`` is given.
        """
        if (
            redshift is None
            and luminosity_distance is None
            and angular_diameter_distance is None
            and proper_distance is None
        ):
            raise ValueError(
                "as_source_spectrum requires an observed flux -- pass "
                "`redshift=` or one of the distance keywords accepted by "
                "as_astropy_model. A SourceSpectrum must be a real per-area "
                "flux, not a rest-frame luminosity."
            )

        model = cls.as_astropy_model(
            x_type="lambda",
            y_type="lambda",
            y_kind="photon",
            wave_unit=u.AA,
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
            **parameters,
        )
        t_cgs = to_cgs_value(t)

        def evaluate(wave: FloatArray) -> FloatResult:
            return model(wave, t_cgs)

        model_class = model_class_from_kernel(
            "_FixedTimeSourceSpectrumModel",
            inputs={"wave": u.AA},
            outputs={"y": synphot_units.PHOTLAM},
            evaluate=evaluate,
        )
        return SourceSpectrum(model_class())

    # -------------------------------------- #
    # Observed Flux Density: F_nu(nu, t)      #
    # -------------------------------------- #
    @classmethod
    def _eval_flux(
        cls,
        nu: FloatArray,
        t: FloatArray,
        redshift: FloatArray,
        luminosity_distance: FloatArray,
        *,
        log_attenuation: FloatArray | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the observed flux density, in cgs units.

        Implements the K-correction-free relation (Hogg 1999):

        .. math::

            F_\nu(\nu, t) = (1+z) \cdot L_\nu\big(\nu(1+z),\ t/(1+z)\big) / (4\pi D_L^2)
                \cdot \exp(\ell(\nu))

        :math:`\nu(1+z)` and :math:`t/(1+z)` convert the observed
        frequency/time to the rest-frame values :meth:`_eval` expects; the
        :math:`(1+z)` prefactor accounts for observed-bandwidth compression.
        :math:`\ell(\nu)` is `log_attenuation`, an observed-frame effect
        (e.g. Milky Way foreground dust) applied here rather than inside
        :meth:`_eval`, so it does not also leak into
        :meth:`_eval_bolometric`/:meth:`as_astropy_model`'s (undiluted) normalization.

        Parameters
        ----------
        nu, t
            Observed frequency (Hz) and time since explosion (s).
        redshift
            Cosmological redshift, dimensionless.
        luminosity_distance
            Luminosity distance, in cm.
        log_attenuation
            Natural log of an observed-frame multiplicative attenuation
            (e.g. the log of Milky Way dust transmission), added directly to
            the log flux. ``None`` (the default) applies no attenuation.
        **parameters
            This model's parameter values, in cgs units, broadcastable
            against ``nu``/``t``/``redshift``/``luminosity_distance``.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz.
        """
        log_flux = (
            np.log1p(redshift)
            + cls._eval(nu * (1.0 + redshift), t / (1.0 + redshift), **parameters)
            - np.log(4.0 * np.pi)
            - 2.0 * np.log(luminosity_distance)
        )

        if log_attenuation is not None:
            log_flux = log_flux + log_attenuation

        return log_flux

    @classmethod
    def flux_log_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the observed flux density, taking and returning plain cgs numbers.

        Unlike :meth:`flux_log`, ``redshift``/``luminosity_distance`` are not
        resolved from a cosmology here -- pass already-computed values
        directly (e.g. a precomputed per-event grid).

        Parameters
        ----------
        nu, t
            Observed frequency (Hz) and time since explosion (s).
        redshift
            Cosmological redshift, dimensionless.
        luminosity_distance
            Luminosity distance, in cm.
        log_attenuation
            Natural log of an observed-frame multiplicative attenuation,
            added directly to the log flux. Must already broadcast against
            the natural output shape of this call.
        **parameters
            This model's parameter values, in cgs units.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz. A scalar is
            returned if the result is 0-dimensional.
        """
        nu_arr = np.asarray(nu, dtype=np.float64)
        t_arr = np.asarray(t, dtype=np.float64)
        redshift_arr = np.asarray(redshift, dtype=np.float64)
        luminosity_distance_arr = np.asarray(luminosity_distance, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval_flux(
            nu_arr,
            t_arr,
            redshift_arr,
            luminosity_distance_arr,
            log_attenuation=None
            if log_attenuation is None
            else np.asarray(log_attenuation, dtype=np.float64),
            **cgs_parameters,
        )
        return result.item() if result.ndim == 0 else result

    @classmethod
    def flux_log(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the observed flux density, given physical-unit inputs.

        Parameters
        ----------
        nu
            Observed frequency.
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            Exactly one of ``redshift`` or the three distance keywords must
            be given; the rest are derived from it using ``cosmology`` (see
            :func:`~m4opt.models._cosmology.resolve_cosmological_distances`).
            ``cosmology`` defaults to that function's configured default.
        log_attenuation
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        numpy.ndarray or float
            The natural log of :math:`F_\nu`, in erg/s/cm^2/Hz.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.flux_log_cgs(
            nu.cgs.value,
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            log_attenuation=log_attenuation,
            **cgs_parameters,
        )

    @classmethod
    def flux_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """Observed flux density, taking and returning plain cgs numbers. See :meth:`flux_log_cgs`."""
        return np.exp(
            cls.flux_log_cgs(
                nu,
                t,
                redshift,
                luminosity_distance,
                log_attenuation=log_attenuation,
                **parameters,
            )
        )

    @classmethod
    def flux(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the observed flux density at the given frequency and time.

        Parameters
        ----------
        nu
            Observed frequency.
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            See :meth:`flux_log`.
        log_attenuation
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`F_\nu(\nu, t)`, in erg/s/cm^2/Hz.
        """
        return (
            np.exp(
                cls.flux_log(
                    nu,
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    log_attenuation=log_attenuation,
                    **parameters,
                )
            )
            * _SPEC_FLUX_UNIT
        )

    # -------------------------------------- #
    # Observed Bolometric Flux: F_bol(t)      #
    # -------------------------------------- #
    @classmethod
    def _eval_flux_bolometric(
        cls,
        t: FloatArray,
        redshift: FloatArray,
        luminosity_distance: FloatArray,
        **parameters: CGSParameterValue,
    ) -> FloatArray:
        r"""
        Evaluate the natural log of the observed bolometric flux, in cgs units.

        .. math::

            F_\mathrm{bol}(t) = L_\mathrm{bol}(t/(1+z)) / (4\pi D_L^2)

        No :math:`(1+z)` prefactor appears here: integrating
        :meth:`_eval_flux`'s :math:`F_\nu(\nu_\mathrm{obs})` over all
        observed frequency and substituting :math:`\nu_\mathrm{emit} =
        \nu_\mathrm{obs}(1+z)` makes that factor cancel exactly against the
        Jacobian of the substitution.
        """
        return (
            cls._eval_bolometric(t / (1.0 + redshift), **parameters)
            - np.log(4.0 * np.pi)
            - 2.0 * np.log(luminosity_distance)
        )

    @classmethod
    def flux_bolometric_log_cgs(
        cls,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """
        Natural log of the observed bolometric flux, taking and returning plain cgs numbers.

        Parameters
        ----------
        t
            Time since explosion, in seconds.
        redshift
            Cosmological redshift, dimensionless.
        luminosity_distance
            Luminosity distance, in cm.
        **parameters
            This model's parameter values, in cgs units; must already
            broadcast against ``t``.

        Returns
        -------
        numpy.ndarray or float
            The natural log of the observed bolometric flux, in erg/s/cm^2.
            A scalar is returned if the result is 0-dimensional.
        """
        t_arr = np.asarray(t, dtype=np.float64)
        redshift_arr = np.asarray(redshift, dtype=np.float64)
        luminosity_distance_arr = np.asarray(luminosity_distance, dtype=np.float64)
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in parameters.items()
        }

        result = cls._eval_flux_bolometric(
            t_arr, redshift_arr, luminosity_distance_arr, **cgs_parameters
        )

        return result.item() if result.ndim == 0 else result

    @classmethod
    def flux_bolometric_log(
        cls,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        """Natural log of the observed bolometric flux, given physical-unit inputs. See :meth:`flux_log`."""
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )

        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.flux_bolometric_log_cgs(
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            **cgs_parameters,
        )

    @classmethod
    def flux_bolometric_cgs(
        cls,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """Observed bolometric flux, taking and returning plain cgs numbers. See :meth:`flux_log_cgs`."""
        return np.exp(
            cls.flux_bolometric_log_cgs(t, redshift, luminosity_distance, **parameters)
        )

    @classmethod
    def flux_bolometric(
        cls,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the observed bolometric flux at the given time.

        Parameters
        ----------
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            See :meth:`flux_log`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`F_\mathrm{bol}(t)`, in erg/s/cm^2.
        """
        return (
            np.exp(
                cls.flux_bolometric_log(
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    **parameters,
                )
            )
            * _BOL_FLUX_UNIT
        )

    # -------------------------------------- #
    # Observed Band-Averaged Flux: F_band(t)  #
    # -------------------------------------- #
    @classmethod
    def flux_band_log_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Natural log of the throughput-weighted mean flux density over a band, plain cgs numbers.

        .. math::

            \log \bar{F}_\nu = \log \frac{\int F_\nu(\nu, t) \cdot T(\nu)\,d\nu}{\int T(\nu)\,d\nu}

        using :meth:`flux_cgs`'s (redshifted, distance-diluted) :math:`F_\nu`,
        integrated over the observed-frame frequency grid ``nu`` and bandpass
        response ``throughput`` by trapezoidal quadrature. Dividing by the
        integrated throughput keeps the result dimensionally a flux density,
        directly comparable to :meth:`flux_cgs`.

        Because this method integrates away a frequency axis, ``t``,
        ``redshift``, ``luminosity_distance``, and every parameter are each
        given one trailing axis internally so they broadcast against the
        ``nu`` grid; any shape you would otherwise pass unchanged to
        :meth:`flux_log_cgs` still works here.

        Parameters
        ----------
        nu
            Observed frequency grid to integrate over, in Hz, shape ``(K,)``.
            Need not be sorted.
        throughput
            Dimensionless bandpass response at each ``nu`` sample, shape
            ``(K,)``.
        t
            Observed time since explosion, in seconds, any shape.
        redshift
            Cosmological redshift, dimensionless, any shape.
        luminosity_distance
            Luminosity distance, in cm, any shape.
        log_attenuation
            Natural log of an observed-frame multiplicative attenuation,
            sampled at the same ``nu`` grid (its last axis must have length
            ``K``, in ``nu``'s original, pre-sort order). Any leading axes
            broadcast against ``t``/``redshift``/``luminosity_distance``/the
            parameters.
        **parameters
            This model's parameter values, in cgs units, any shape.

        Returns
        -------
        numpy.ndarray
            The natural log of :math:`\bar{F}_\nu`, in erg/s/cm^2/Hz, with
            the broadcast shape of ``t``/``redshift``/``luminosity_distance``/
            the parameters (the frequency axis is integrated away).
        """
        nu = np.asarray(nu, dtype=np.float64)
        throughput = np.asarray(throughput, dtype=np.float64)
        order = np.argsort(nu)
        nu_sorted = nu[order]
        throughput_sorted = throughput[order]

        band_ready_parameters: dict[str, CGSParameterValue] = {
            name: np.asarray(value, dtype=np.float64)[..., np.newaxis]
            for name, value in parameters.items()
        }

        log_flux_density = cls._eval_flux(
            nu_sorted,
            np.asarray(t, dtype=np.float64)[..., np.newaxis],
            np.asarray(redshift, dtype=np.float64)[..., np.newaxis],
            np.asarray(luminosity_distance, dtype=np.float64)[..., np.newaxis],
            log_attenuation=(
                None
                if log_attenuation is None
                else np.asarray(log_attenuation, dtype=np.float64)[..., order]
            ),
            **band_ready_parameters,
        )

        numerator = np.trapezoid(
            np.exp(log_flux_density) * throughput_sorted, nu_sorted, axis=-1
        )
        denominator = np.trapezoid(throughput_sorted, nu_sorted)

        return np.log(numerator / denominator)

    @classmethod
    def flux_band_log(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> FloatResult:
        """Natural log of the band-averaged observed flux density, given physical-unit inputs. See :meth:`flux_log`."""
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return cls.flux_band_log_cgs(
            nu.cgs.value,
            throughput,
            t.cgs.value,
            np.asarray(distances["redshift"], dtype=np.float64),
            distances["luminosity_distance"].cgs.value,
            log_attenuation=log_attenuation,
            **cgs_parameters,
        )

    @classmethod
    def flux_band_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """Band-averaged observed flux density, taking and returning plain cgs numbers. See :meth:`flux_band_log_cgs`."""
        return np.exp(
            cls.flux_band_log_cgs(
                nu,
                throughput,
                t,
                redshift,
                luminosity_distance,
                log_attenuation=log_attenuation,
                **parameters,
            )
        )

    @classmethod
    def flux_band(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        r"""
        Evaluate the throughput-weighted mean observed flux density over a band.

        Parameters
        ----------
        nu
            Observed frequency grid to integrate over. Need not be sorted.
        throughput
            Dimensionless bandpass response at each ``nu`` sample.
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            See :meth:`flux_log`.
        log_attenuation
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`\bar{F}_\nu`, in erg/s/cm^2/Hz.
        """
        return (
            np.exp(
                cls.flux_band_log(
                    nu,
                    throughput,
                    t,
                    redshift=redshift,
                    luminosity_distance=luminosity_distance,
                    angular_diameter_distance=angular_diameter_distance,
                    proper_distance=proper_distance,
                    cosmology=cosmology,
                    log_attenuation=log_attenuation,
                    **parameters,
                )
            )
            * _SPEC_FLUX_UNIT
        )

    # -------------------------------------- #
    # Apparent AB Magnitudes                  #
    # -------------------------------------- #
    # A magnitude already is a logarithmic quantity, so there is no separate
    # "log" form here -- just a bare-float `*_cgs` form and a unitful form
    # returning an `astropy.units.Magnitude` (`u.ABmag`).
    @classmethod
    def mag_cgs(
        cls,
        nu: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        r"""
        Apparent AB magnitude: :math:`m_\mathrm{AB} = -2.5 \log_{10}(F_\nu / F_{\mathrm{AB},0})`.

        :math:`F_\nu` is :meth:`flux_cgs`'s observed flux density;
        :math:`F_{\mathrm{AB},0} = 3631` Jy. See :meth:`flux_log_cgs` for the
        meaning of ``log_attenuation``.
        """
        F_nu = cls.flux_cgs(
            nu,
            t,
            redshift,
            luminosity_distance,
            log_attenuation=log_attenuation,
            **parameters,
        )

        return -2.5 * np.log10(F_nu / AB_MAG_ZERO_POINT)

    @classmethod
    def mag(
        cls,
        nu: Quantity,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        """
        Evaluate the apparent AB magnitude at the given frequency and time.

        Parameters
        ----------
        nu
            Observed frequency.
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            See :meth:`flux_log`.
        log_attenuation
            See :meth:`flux_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return (
            cls.mag_cgs(
                nu.cgs.value,
                t.cgs.value,
                np.asarray(distances["redshift"], dtype=np.float64),
                distances["luminosity_distance"].cgs.value,
                log_attenuation=log_attenuation,
                **cgs_parameters,
            )
            * u.ABmag
        )

    @classmethod
    def mag_band_cgs(
        cls,
        nu: NumericalInput,
        throughput: NumericalInput,
        t: NumericalInput,
        redshift: NumericalInput,
        luminosity_distance: NumericalInput,
        *,
        log_attenuation: NumericalInput | None = None,
        **parameters: CGSParameterValue,
    ) -> FloatResult:
        """
        Apparent AB magnitude of the band-averaged flux density. See :meth:`flux_band_cgs`/:meth:`mag_cgs`.

        Broadcasting (including ``log_attenuation``'s) follows :meth:`flux_band_cgs`'s rules exactly.
        """
        F_nu = cls.flux_band_cgs(
            nu,
            throughput,
            t,
            redshift,
            luminosity_distance,
            log_attenuation=log_attenuation,
            **parameters,
        )

        return -2.5 * np.log10(F_nu / AB_MAG_ZERO_POINT)

    @classmethod
    def mag_band(
        cls,
        nu: Quantity,
        throughput: NumericalInput,
        t: Quantity,
        *,
        redshift: PhysicalInput | None = None,
        luminosity_distance: Quantity | None = None,
        angular_diameter_distance: Quantity | None = None,
        proper_distance: Quantity | None = None,
        cosmology: FLRW | None = None,
        log_attenuation: NumericalInput | None = None,
        **parameters: ParameterValue,
    ) -> Quantity:
        """
        Evaluate the apparent AB magnitude of the band-averaged flux density.

        Parameters
        ----------
        nu
            Observed frequency grid to integrate over. Need not be sorted.
        throughput
            Dimensionless bandpass response at each ``nu`` sample.
        t
            Observed time since explosion.
        redshift, luminosity_distance, angular_diameter_distance, proper_distance, cosmology
            See :meth:`flux_log`.
        log_attenuation
            See :meth:`flux_band_log_cgs`.
        **parameters
            This model's parameter values. See :meth:`eval_log_cgs`.

        Returns
        -------
        ~astropy.units.Quantity
            The apparent AB magnitude, as an :attr:`~astropy.units.ABmag` Quantity.
        """
        if not isinstance(nu, Quantity) or nu.unit.physical_type != "frequency":
            raise TypeError("`nu` must be an astropy Quantity with frequency units.")
        if not isinstance(t, Quantity) or t.unit.physical_type != "time":
            raise TypeError("`t` must be an astropy Quantity with time units.")

        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        cgs_parameters: dict[str, CGSParameterValue] = {
            name: to_cgs_value(value) for name, value in parameters.items()
        }

        return (
            cls.mag_band_cgs(
                nu.cgs.value,
                throughput,
                t.cgs.value,
                np.asarray(distances["redshift"], dtype=np.float64),
                distances["luminosity_distance"].cgs.value,
                log_attenuation=log_attenuation,
                **cgs_parameters,
            )
            * u.ABmag
        )

    # -------------------------------------- #
    # Simulation                              #
    # -------------------------------------- #
    def simulate(
        self, nu: Quantity, t: Quantity, size: int = 1, *, rng: RNGInput = None
    ) -> Quantity:
        r"""
        Draw random parameter realizations and evaluate the model at the given frequency and time.

        Equivalent to ``self.eval(nu, t, **self.sample_parameters(size=size, rng=rng))``.

        Parameters
        ----------
        nu
            Frequency at which to evaluate the model. See :meth:`eval`.
        t
            Time since explosion. See :meth:`eval`.
        size
            Number of realizations to draw.
        rng
            Random-number source, forwarded to :meth:`sample_parameters`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L_\nu(\nu, t)`, in erg/s/Hz. :meth:`sample_parameters`
            always returns array-valued parameters, even for ``size=1``, so
            the batch axis is never squeezed away here.

        See Also
        --------
        eval : The underlying evaluation.
        sample_parameters : The underlying sampling.
        """
        return self.eval(nu, t, **self.sample_parameters(size=size, rng=rng))


class ComposedSpectralModel(SpectralModel):
    r"""
    A :class:`SpectralModel` built by pairing a :class:`Lightcurve` with a :class:`Spectrum`.

    .. math::

        L_\nu(\nu, t) = L_\mathrm{bol}(t) \cdot \frac{S(\nu)}{\int S(\nu')\,d\nu'}

    where :math:`L_\mathrm{bol}(t)` is :attr:`_LIGHTCURVE_CLASS`'s bolometric
    luminosity and the fraction is :attr:`_SPECTRUM_CLASS`'s shape,
    normalized (via :meth:`Spectrum.eval_normalization`) to integrate to 1
    over :math:`\nu`. Because both halves are already exact on their own,
    :meth:`_eval_bolometric` and :meth:`_eval_spectrum` are closed-form
    combinations of the two components' own primitives -- unlike
    :class:`SpectralModel`'s generic defaults, neither ever falls back to
    numerical integration over frequency here.

    A concrete SED is defined just by naming the two component classes:

    .. code-block:: python

        class MySED(ComposedSpectralModel):
            _LIGHTCURVE_CLASS = FREDLightcurve
            _SPECTRUM_CLASS = BlackbodySpectrum

    At class-definition time, :meth:`__init_subclass__` merges
    ``_LIGHTCURVE_CLASS._DEFAULT_PARAMETERS`` and
    ``_SPECTRUM_CLASS._DEFAULT_PARAMETERS`` into this subclass's own
    :attr:`~SpectralModel._DEFAULT_PARAMETERS` (any entries the subclass
    declares directly itself win, as overrides on top of that merge) -- so
    from then on, ``MySED`` behaves exactly like any other
    :class:`SpectralModel` subclass: ``MySED()`` or ``MySED(amplitude=...,
    temperature=...)``, one flat parameter namespace. Every parameter name
    must be unique across the two component classes, checked at
    class-definition time.

    Because every :class:`SpectralModel` method is a classmethod operating
    purely on ``**parameters`` (see the module docstring), no instance-level
    wiring of components is needed: :meth:`_eval_bolometric`,
    :meth:`_eval_spectrum`, and :meth:`_eval` simply split the incoming
    ``parameters`` dict by name and call :attr:`_LIGHTCURVE_CLASS`'s/
    :attr:`_SPECTRUM_CLASS`'s own classmethods directly.

    See Also
    --------
    Lightcurve : The time-only half of this composition.
    Spectrum : The frequency-only half of this composition.
    """

    _LIGHTCURVE_CLASS: ClassVar[type[Lightcurve] | None] = None
    """type of Lightcurve, or None: The bolometric lightcurve class driving this SED's time dependence.

    ``None`` on :class:`ComposedSpectralModel` itself; every concrete
    subclass must set this (and :attr:`_SPECTRUM_CLASS`) directly in its own
    class body.
    """

    _SPECTRUM_CLASS: ClassVar[type[Spectrum] | None] = None
    """type of Spectrum, or None: The spectral shape class driving this SED's frequency dependence."""

    # -------------------------------------- #
    # Subclass Validation                    #
    # -------------------------------------- #
    def __init_subclass__(cls, **kwargs) -> None:
        """Merge the component classes' parameters into :attr:`_DEFAULT_PARAMETERS`."""
        super().__init_subclass__(**kwargs)

        if cls._LIGHTCURVE_CLASS is None or cls._SPECTRUM_CLASS is None:
            return

        if not (
            isinstance(cls._LIGHTCURVE_CLASS, type)
            and issubclass(cls._LIGHTCURVE_CLASS, Lightcurve)
        ):
            raise TypeError(
                f"{cls.__name__}._LIGHTCURVE_CLASS must be a Lightcurve subclass."
            )
        if not (
            isinstance(cls._SPECTRUM_CLASS, type)
            and issubclass(cls._SPECTRUM_CLASS, Spectrum)
        ):
            raise TypeError(
                f"{cls.__name__}._SPECTRUM_CLASS must be a Spectrum subclass."
            )

        overlap = set(cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS) & set(
            cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS
        )
        if overlap:
            raise TypeError(
                f"{cls.__name__}: _LIGHTCURVE_CLASS and _SPECTRUM_CLASS share "
                f"parameter name(s) {sorted(overlap)}; every parameter name must "
                "be unique across the two."
            )

        cls._DEFAULT_PARAMETERS = {
            **cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS,
            **cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS,
            **cls.__dict__.get("_DEFAULT_PARAMETERS", {}),
        }

    # -------------------------------------- #
    # Parameter Splitting                    #
    # -------------------------------------- #
    @classmethod
    def _split_parameters(
        cls, parameters: Mapping[str, CGSParameterValue]
    ) -> tuple[dict[str, CGSParameterValue], dict[str, CGSParameterValue]]:
        """Split a flat ``parameters`` dict into ``(lightcurve_parameters, spectrum_parameters)``."""
        if cls._LIGHTCURVE_CLASS is None or cls._SPECTRUM_CLASS is None:
            raise TypeError(
                f"{cls.__name__} must be subclassed with both _LIGHTCURVE_CLASS "
                "and _SPECTRUM_CLASS set before it can be evaluated."
            )

        return (
            {
                name: parameters[name]
                for name in cls._LIGHTCURVE_CLASS._DEFAULT_PARAMETERS
            },
            {
                name: parameters[name]
                for name in cls._SPECTRUM_CLASS._DEFAULT_PARAMETERS
            },
        )

    # -------------------------------------- #
    # Bolometric Luminosity: L_bol(t)         #
    # -------------------------------------- #
    @classmethod
    def _eval_bolometric(
        cls, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        :math:`\log L_\mathrm{bol}(t)`, delegated directly to :attr:`_LIGHTCURVE_CLASS`.

        Exact, not an approximation: unlike
        :meth:`SpectralModel._eval_bolometric`'s generic numerical-quadrature
        fallback, no integration is needed here -- the lightcurve's own
        :meth:`Lightcurve._eval` already *is* the bolometric luminosity, by
        construction.
        """
        lightcurve_parameters, _ = cls._split_parameters(parameters)

        # `_split_parameters` already raises if either class is unset.
        assert cls._LIGHTCURVE_CLASS is not None
        return cls._LIGHTCURVE_CLASS._eval(t, **lightcurve_parameters)

    # -------------------------------------- #
    # Normalized Spectral Shape: S(nu, t)    #
    # -------------------------------------- #
    @classmethod
    def _eval_spectrum(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        :math:`\log S(\nu)`, delegated to :attr:`_SPECTRUM_CLASS` and normalized to integrate to 1 (``t`` is unused).

        Exact, not an approximation: divides out the spectrum's own
        :meth:`Spectrum._eval_normalization` directly, rather than falling
        back to :meth:`SpectralModel._eval_spectrum`'s
        bolometric-integral-subtraction default.
        """
        _, spectrum_parameters = cls._split_parameters(parameters)

        # `_split_parameters` already raises if either class is unset.
        assert cls._SPECTRUM_CLASS is not None
        return cls._SPECTRUM_CLASS._eval(
            nu, **spectrum_parameters
        ) - cls._SPECTRUM_CLASS._eval_normalization(**spectrum_parameters)

    # -------------------------------------- #
    # Spectral Luminosity: L_nu(nu, t)        #
    # -------------------------------------- #
    @classmethod
    def _eval(
        cls, nu: FloatArray, t: FloatArray, **parameters: CGSParameterValue
    ) -> FloatArray:
        r"""
        :math:`\log L_\nu(\nu, t) = \log L_\mathrm{bol}(t) + \log S(\nu)`.

        Since this sums :meth:`_eval_bolometric`'s (``t``-shaped) and
        :meth:`_eval_spectrum`'s (``nu``-shaped) results, ``nu``/``t`` must
        already broadcast against each other the way the caller wants --
        see :meth:`SpectralModel._eval`'s broadcasting contract.
        """
        return cls._eval_bolometric(t, **parameters) + cls._eval_spectrum(
            nu, t, **parameters
        )
