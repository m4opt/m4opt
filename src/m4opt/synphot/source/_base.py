"""Base classes for time-dependent spectral energy distributions (SEDs).

This module provides the :class:`SEDModel`, :class:`LightcurveModel`, and
:class:`SpectrumModel` classes for time-dependent SEDs, bolometric light curves, and spectral shapes, respectively.
Each of these is structured around :class:`astropy.modeling.Model` with additional capabilities.
"""

import copy
from abc import ABC, abstractmethod
from typing import ClassVar

import astropy.units as u
import numpy as np
from astropy.modeling import Model, fix_inputs


class _ModelSamplingMixin:
    """Shared `sample_parameters` for `SEDModel`, `LightcurveModel`, and `Spectrum`.

    This class provides a simple mixin composition for the ``astropy.modeling.Model`` class
    to provide sampling from the existing set of parameters.
    """

    def sample_parameters(self, size=None, rng=None):
        """Draw parameter values: sample any `PriorParameter`, else take the bound value.

        Parameters
        ----------
        size
            Number of samples to draw.
        rng
            Seed or `~numpy.random.Generator`.

        Returns
        -------
        dict
            Parameter name mapped to its sampled or bound value.
        """
        rng = np.random.default_rng(rng)
        samples = {}
        for name in self.param_names:
            parameter = getattr(self, name)
            if parameter.prior is not None and not parameter.fixed:
                value = parameter.prior.rvs(size=size, random_state=rng)
                if parameter.unit is not None:
                    value = value.to(parameter.unit)
            else:
                value = (
                    parameter.quantity
                    if parameter.unit is not None
                    else parameter.value
                )
            samples[name] = value
        return samples


class _ComposedSEDModelMeta(type(Model)):  # type: ignore[misc]
    """Merges ``_LIGHTCURVE``'s and ``_SPECTRUM``'s parameters onto the class.

    This metaclass is designed to preempt the instantiation of the model class so that
    we can compose the parameters from a light curve and a spectrum together into a single
    model. This saves some effort in avoiding having to re-implement common lightcurve / spectrum
    models.
    """

    def __new__(mcls, name, bases, namespace, **kwargs):
        lightcurve_cls = namespace.get("_LIGHTCURVE")
        spectrum_cls = namespace.get("_SPECTRUM")

        if lightcurve_cls is not None and spectrum_cls is not None:
            # abstract subclasses may still not implement, so we allow a pass
            # if they are both none.
            lightcurve_names = list(lightcurve_cls.param_names)
            spectrum_names = list(spectrum_cls.param_names)

            # Check for collisions in the namespace. We need to raise an
            # error if one occurs.
            collisions = set(lightcurve_names) & set(spectrum_names)
            if collisions:
                raise TypeError(
                    f"{name}: _LIGHTCURVE and _SPECTRUM parameter names "
                    f"collide: {sorted(collisions)}"
                )

            # Produce deep copies of the parameters so that we don't get leakage.
            for param_name in lightcurve_names:
                namespace[param_name] = copy.deepcopy(
                    getattr(lightcurve_cls, param_name)
                )
            for param_name in spectrum_names:
                namespace[param_name] = copy.deepcopy(getattr(spectrum_cls, param_name))

            # Determine the spacing of the parameters and generate the evaluation method
            # to place onto the class.
            n_lightcurve_params = len(lightcurve_names)

            def evaluate(freq, time, *parameters):
                lightcurve_values = parameters[:n_lightcurve_params]
                spectrum_values = parameters[n_lightcurve_params:]
                return lightcurve_cls.evaluate(
                    time, *lightcurve_values
                ) * spectrum_cls.evaluate(freq, *spectrum_values)

            namespace["evaluate"] = staticmethod(evaluate)

        return super().__new__(mcls, name, bases, namespace, **kwargs)


class SEDModel(Model, _ModelSamplingMixin, ABC):
    """A time-dependent SED model.

    :class:`SEDModel` is the base class of all samplable spectral models in ``m4opt``. It is
    effectively an astropy :class:`astropy.modeling.Model` object with two parameters: ``freq`` and ``time``, and one
    output ``lnu``. The model is expected to be a function of both frequency and time,
    and the output is the spectral luminosity density :math:`L_\\nu(\\nu, t)`.
    """

    n_inputs = 2
    n_outputs = 1

    inputs = ("freq", "time")
    outputs = ("lnu",)

    input_units: ClassVar = {
        "freq": u.Hz,
        "time": u.day,
    }

    input_units_equivalencies: ClassVar = {
        "freq": u.spectral(),
    }

    return_units: ClassVar = {
        "lnu": u.erg / (u.s * u.Hz),
    }

    @staticmethod
    @abstractmethod
    def evaluate(freq, time, *parameters):
        """Evaluate L_nu(nu, t)."""
        raise NotImplementedError

    def get_spectrum_model(self, time, source_redshift=0.0):
        """Fix ``time`` and return a model of frequency alone.

        Parameters
        ----------
        time
            Observed time since the reference epoch.
        source_redshift
            Redshift used to convert ``time`` to the rest frame.

        Returns
        -------
        ~astropy.modeling.Model
            Callable as ``model(freq)``.
        """
        rest_frame_time = time / (1 + source_redshift)
        return fix_inputs(self, {"time": rest_frame_time})

    def get_lightcurve_model(self, frequency, source_redshift=0.0):
        """Fix ``frequency`` and return a model of time alone.

        Parameters
        ----------
        frequency
            Observed frequency.
        source_redshift
            Redshift used to convert ``frequency`` to the rest frame.

        Returns
        -------
        ~astropy.modeling.Model
            Callable as ``model(time)``.
        """
        rest_frame_frequency = frequency * (1 + source_redshift)
        return fix_inputs(self, {"freq": rest_frame_frequency})


class LightcurveModel(Model, _ModelSamplingMixin, ABC):
    """A bolometric light curve shape: a function of time alone."""

    n_inputs = 1
    n_outputs = 1

    inputs = ("time",)
    outputs = ("l",)

    input_units: ClassVar = {"time": u.day}

    @staticmethod
    @abstractmethod
    def evaluate(time, *parameters):
        """Evaluate L(t)."""
        raise NotImplementedError


class Spectrum(Model, _ModelSamplingMixin, ABC):
    """A spectral shape: a function of frequency alone."""

    n_inputs = 1
    n_outputs = 1

    inputs = ("freq",)
    outputs = ("s",)

    input_units: ClassVar = {"freq": u.Hz}
    input_units_equivalencies: ClassVar = {"freq": u.spectral()}

    @staticmethod
    @abstractmethod
    def evaluate(freq, *parameters):
        """Evaluate S(nu)."""
        raise NotImplementedError


class ComposedSEDModel(SEDModel, metaclass=_ComposedSEDModelMeta):
    """A SEDModel built as ``lightcurve(t) * spectrum(nu)``.

    Set ``_LIGHTCURVE``/``_SPECTRUM`` to a `LightcurveModel`/`Spectrum`
    subclass; their parameters get merged in, so instances have one flat
    parameter list, e.g. ``VillarCoolingBlackbodySED(t_rise=..., temperature0=...)``.
    ``_LIGHTCURVE`` and ``_SPECTRUM`` must not share parameter names.
    """

    _LIGHTCURVE: ClassVar[type[LightcurveModel]]
    _SPECTRUM: ClassVar[type[Spectrum]]
