"""
Numerical coercion and unit-conversion utilities for :mod:`m4opt.models`.

This module provides small helpers for normalizing user-facing numerical
inputs before they are passed to model calculations. Inputs may be ordinary
NumPy-compatible values or :class:`astropy.units.Quantity` objects; outputs
are plain NumPy arrays with units removed.

Unit-aware helpers explicitly convert quantities before stripping their units.
Unitless numerical inputs are assumed to already be expressed in the requested
units.
"""

from collections.abc import Callable

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.modeling import Model
from numpy.typing import DTypeLike, NDArray

from m4opt.models._typing import FloatArray, FloatResult, PhysicalInput, UnitLike

# ------------------------------------------ #
# Internal Unit Conventions                  #
# ------------------------------------------ #
# To avoid explicit / lengthy units in the code, we define a set of standard units for
# outputs from the m4opt.models library.
_SPEC_FLUX_UNIT: u.Unit = u.erg / u.s / u.Hz / u.cm**2
_BOL_FLUX_UNIT: u.Unit = u.erg / u.s / u.cm**2
_SPEC_LUM_UNIT: u.Unit = u.erg / u.s / u.Hz
_BOL_LUM_UNIT: u.Unit = u.erg / u.s

# The SED shape unit is the unit used to parameterize the shape of a raw spectrum.
_SED_SHAPE_UNIT: u.Unit = u.Hz**-1

# ------------------------------------------ #
# Zero-Points and Constants                  #
# ------------------------------------------ #
AB_MAG_ZERO_POINT: float = 3631e-23
"""float: The CGS zero-point flux in the AB mag system."""

H_CGS: float = const.h.cgs.value
"""float: The Planck constant, in erg s."""

# ------------------------------------------ #
# Unit Coercion Functions                    #
# ------------------------------------------ #
# These functions are each concerned with manipulating numpy / astropy arrays and quantities
# into one another and are used heavily in the m4opt.models module to coerce unit-bearing user
# input to raw numerical internal arrays.


def ensure_numpy_array(
    value: PhysicalInput,
    dtype: DTypeLike | None = None,
) -> NDArray:
    """
    Coerce a numerical input to a NumPy array.

    Astropy quantities are accepted only when they are dimensionless. Their
    units are stripped before conversion. All other inputs are passed directly
    to :func:`numpy.asarray`.

    The function follows normal NumPy coercion semantics: scalar inputs become
    zero-dimensional arrays, existing arrays are reused when possible, and the
    dtype is preserved unless an explicit ``dtype`` is requested.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert. Quantity inputs must be convertible to
        :attr:`astropy.units.dimensionless_unscaled`.
    dtype : numpy.dtype-like, optional
        Desired dtype of the returned array. If omitted, NumPy infers or
        preserves the dtype.

    Returns
    -------
    numpy.ndarray
        NumPy representation of ``value``.

    Raises
    ------
    astropy.units.UnitConversionError
        If ``value`` is a dimensional quantity.

    Examples
    --------
    >>> ensure_numpy_array([1, 2, 3])
    array([1, 2, 3])

    >>> ensure_numpy_array([1, 2, 3], dtype=np.float64)
    array([1., 2., 3.])

    >>> ensure_numpy_array(2.0 * u.dimensionless_unscaled)
    array(2.)
    """
    if isinstance(value, u.Quantity):
        value = value.to_value(u.dimensionless_unscaled)

    return np.asarray(value, dtype=dtype)


def to_cgs_value(value: PhysicalInput) -> FloatArray:
    """
    Convert a numerical input to unit-stripped CGS values.

    Quantity inputs are converted to their corresponding CGS representation
    before their units are removed. Unitless inputs are assumed to already be
    expressed in CGS units.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert.

    Returns
    -------
    numpy.ndarray
        Numerical values expressed in the appropriate CGS units, with dtype
        ``float64``.

    Examples
    --------
    >>> to_cgs_value(1.0 * u.km)
    array(100000.)

    >>> to_cgs_value([1.0, 2.0] * u.kg)
    array([1000., 2000.])

    >>> to_cgs_value([1.0, 2.0])
    array([1., 2.])
    """
    if isinstance(value, u.Quantity):
        value = value.cgs.value

    return np.asarray(value, dtype=np.float64)


def ensure_in_units(
    value: PhysicalInput,
    unit: UnitLike,
) -> FloatArray:
    """
    Convert a numerical input to unit-stripped values in a specified unit.

    Quantity inputs are converted to ``unit`` before their units are removed.
    Unitless inputs are assumed to already be expressed in ``unit`` and are
    therefore unchanged apart from conversion to a ``float64`` NumPy array.

    If ``unit`` is ``None``, the requested unit is interpreted as
    :attr:`astropy.units.dimensionless_unscaled`.

    Parameters
    ----------
    value : array_like or ~astropy.units.Quantity
        Numerical input to convert.
    unit : str, ~astropy.units.UnitBase, or None
        Unit in which the returned numerical values should be expressed.
        ``None`` denotes a dimensionless value.

    Returns
    -------
    numpy.ndarray
        Unit-stripped numerical values expressed in ``unit``, with dtype
        ``float64``.

    Raises
    ------
    ValueError
        If ``unit`` cannot be interpreted as a valid Astropy unit.
    astropy.units.UnitConversionError
        If ``value`` is a quantity incompatible with ``unit``.

    Examples
    --------
    >>> ensure_in_units(1500.0 * u.m, u.km)
    array(1.5)

    >>> ensure_in_units([1.0, 2.0] * u.MHz, "Hz")
    array([1000000., 2000000.])

    >>> ensure_in_units([1.0, 2.0], u.s)
    array([1., 2.])
    """
    unit = u.dimensionless_unscaled if unit is None else u.Unit(unit)

    if isinstance(value, u.Quantity):
        value = value.to_value(unit)

    return np.asarray(value, dtype=np.float64)


def hz_per_unit(unit: UnitLike, *, is_wavelength: bool) -> float:
    r"""
    The coefficient converting a bare number in ``unit`` to a frequency in Hz.

    Resolved once, up front -- not on every element of every model
    evaluation -- so that letting a model's spectral axis be expressed in
    an arbitrary (self-consistent) unit costs one extra multiply or divide
    per evaluation, not a :class:`~astropy.units.Quantity` conversion in
    the hot loop.

    Parameters
    ----------
    unit : str or ~astropy.units.UnitBase
        A wavelength unit if ``is_wavelength``, otherwise a frequency unit.
    is_wavelength : bool
        Whether ``unit`` is a wavelength unit (``True``) or a frequency
        unit (``False``).

    Returns
    -------
    float
        If ``is_wavelength`` is ``False``, :math:`\nu_\mathrm{Hz} =
        \mathrm{coefficient} \times x`; if ``True``, :math:`\nu_\mathrm{Hz}
        = \mathrm{coefficient} / x` (frequency is inversely, not linearly,
        related to wavelength). Either way, this is that coefficient.

    Raises
    ------
    ValueError
        If ``unit`` is not a wavelength/frequency unit matching
        ``is_wavelength``.

    Examples
    --------
    >>> hz_per_unit(u.THz, is_wavelength=False)
    1000000000000.0
    """
    resolved = u.Unit(unit)
    expected_type = "length" if is_wavelength else "frequency"
    if resolved.physical_type != expected_type:
        kind = "wavelength" if is_wavelength else "frequency"
        raise ValueError(f"{unit!r} is not a {kind} unit.")

    return resolved.to(u.Hz, equivalencies=u.spectral())


# ------------------------------------------ #
# Astropy Model Construction                 #
# ------------------------------------------ #
# At the m4opt.synphot level, operations are performed on Synphot / Astropy Model objects,
# which are not immediately compatible with the machinery of the SpectralModel class. This
# helper lets a caller wrap a plain broadcasting kernel as such a Model on demand.


def model_class_from_kernel(
    name: str,
    inputs: dict[str, u.UnitBase],
    outputs: dict[str, u.UnitBase],
    evaluate: Callable[..., FloatResult],
) -> type[Model]:
    """
    Dynamically build a parameterless :class:`~astropy.modeling.Model` subclass around a plain broadcasting kernel.

    Deliberately does not use astropy's formal :class:`~astropy.modeling.Parameter`
    machinery -- any state ``evaluate`` needs (e.g. a model's SED
    parameter values, batched or not) must already be bound into it by
    the caller (typically via closure). This is the same pattern used by
    ``m4opt.synphot._extrinsic.ScaleFactor``: because no ``Parameter`` is
    involved, :mod:`synphot`'s ``n_models=1`` restriction never applies,
    and any leading batch axes on the bound state broadcast straight
    through :meth:`~astropy.modeling.Model.evaluate` against whatever
    shape this model is called with.

    Parameters
    ----------
    name
        Class name for the generated :class:`~astropy.modeling.Model` subclass.
    inputs
        ``{name: unit}`` for each positional input, in call order.
    outputs
        ``{name: unit}`` for this model's output. Exactly one entry.
    evaluate
        ``evaluate(*input_values) -> FloatResult``, operating on
        unit-stripped, broadcastable NumPy arrays -- e.g. a closure over
        one of a :class:`~m4opt.models.core.SpectralModel` subclass's
        ``*_cgs`` classmethods. Bound directly as a ``staticmethod`` on
        the generated class (no ``self``).

    Returns
    -------
    type
        A fresh :class:`~astropy.modeling.Model` subclass (not an
        instance) with ``n_inputs``/``n_outputs``/``inputs``/``outputs``/
        ``input_units``/``return_units`` populated from ``inputs``/``outputs``.
    """
    if len(outputs) != 1:
        raise NotImplementedError("Only single-output models are currently supported.")
    (output_name,) = outputs
    input_units = tuple(inputs.values())

    def _strip_units(*values: FloatArray) -> FloatResult:
        # `Model.__call__` converts a Quantity input to the declared unit
        # but -- unlike a bare-number call -- does *not* strip it down to a
        # plain array before handing it to `evaluate`; without this, a
        # Quantity would silently ride along through `evaluate`'s kernel
        # arithmetic, corrupting whatever unit `_process_output_units`
        # later tries to attach to the result. Bare-number calls pass
        # through `to_value` untouched (it's a no-op for non-Quantities).
        return evaluate(
            *(
                value.to_value(unit) if isinstance(value, u.Quantity) else value
                for value, unit in zip(values, input_units)
            )
        )

    def _prepare_outputs(self, broadcasted_shapes, *outputs, **kwargs):
        # `evaluate` already returns whatever shape its bound batch state
        # (broadcast against the call's `x`/`t`) actually produces --
        # unlike a normal Parameter-based model, astropy's shape inference
        # here only sees the *declared* inputs, not any batch axes bound
        # into `evaluate` via closure. The default `prepare_outputs` still
        # applies correctly for the common case (an ordinary, non-batched
        # call, where it turns a size-1 array into a proper scalar); it
        # only needs to be skipped when the closure-bound batch shape isn't
        # visible in the declared inputs' broadcast shape, which makes its
        # unguarded `output.item()`/`.reshape()` raise.
        try:
            return self._prepare_outputs_single_model(outputs, broadcasted_shapes)
        except ValueError:
            return outputs

    return type(
        name,
        (Model,),
        {
            "n_inputs": len(inputs),
            "n_outputs": 1,
            "inputs": tuple(inputs),
            "outputs": (output_name,),
            "input_units": dict(inputs),
            "return_units": dict(outputs),
            # synphot -- and any other caller passing bare floats -- expects
            # them to be interpreted in `inputs`' declared units; mirror
            # `astropy.modeling.physical_models.BlackBody`.
            "_input_units_allow_dimensionless": True,
            "evaluate": staticmethod(_strip_units),
            "prepare_outputs": _prepare_outputs,
        },
    )
