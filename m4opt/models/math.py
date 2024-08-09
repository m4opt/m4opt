from contextlib import contextmanager
from itertools import chain
from typing import Optional, Union

import numpy as np
from astropy.modeling import CompoundModel, Model
from astropy.units.quantity import Quantity

try:
    from numpy import trapezoid
except ImportError:
    # FIXME: remove when we require Numpy >= 2.0.0
    from numpy import trapz as trapezoid
import portion.interval
from scipy.integrate import quad_vec


def _unwrap_scalar(array):
    if isinstance(array, np.generic) and np.isscalar(array):
        array = array.item()
    return array


def _map_interval(func):
    """Apply a function to the bounds of an interval."""

    def wrapper(interval):
        lower = func(interval.lower)
        upper = func(interval.upper)
        if lower > upper:
            lower, upper = upper, lower
        return portion.open(lower, upper)

    return wrapper


def _to_open_interval(interval: portion.Interval) -> portion.Interval:
    """Convert any interval to an open interval."""
    return portion.open(interval.lower, interval.upper)


@contextmanager
def _with_portion_float_inf():
    """Temporarily set infinity used by the portion library to `float('inf')`.

    `float('inf')` is comparable with any Astropy quantity, but `portion.inf`
    is not. The portion uses its representation of infinity internally in order
    to calculate complement sets.
    """
    tmp = portion.interval.inf
    portion.interval.inf = float("inf")
    try:
        yield
    finally:
        portion.interval.inf = tmp


def _get_1d_units_for_dict(units: dict) -> Union[Quantity, float]:
    if units:
        ((_, units),) = units.items()
    if not units:
        units = 1
    return units


@_with_portion_float_inf()
def _get_intervals(model: Model) -> portion.Interval:
    """Calculate the model breakpoints for a 1D Model by combining
    the bounding box endpoints of the submodels

    Parameters
    ----------
    model : :class:`astropy.modeling.Model`
        The model.

    Returns
    -------
    points : list, float
        Model breakpoints to facilitate model integration, sorted from low
        to high. The range for integration will be from the first to the last
        entry in the list.

    Notes
    -----
    Models without bounding boxes are valid in full domain, so breakpoints are
    given as +/- `numpy.inf`. However, infinities will be discarded if
    at least one submodel has a bounding box.

    Examples
    --------

    >>> from astropy.modeling import models
    >>> from astropy import units as u
    >>> from m4opt.models.math import _get_intervals

    For a simple, bounded model, `_get_intervals` returns a single interval:

    >>> _get_intervals(models.Box1D(width=3))
    (-1.5,1.5)

    This also works with models that have dimensionful input:

    >>> _get_intervals(models.Box1D(width=3*u.m))
    (<Quantity -1.5 m>,<Quantity 1.5 m>)

    For unbounded models, `_get_intervals` returns an infinite interval:

    >>> _get_intervals(models.Exponential1D())
    (-inf,inf)
    >>> _get_intervals(models.Exponential1D(tau=1*u.m))
    (<Quantity -inf m>,<Quantity inf m>)

    Compound models are supported too. Supported operations include function
    composition (`|`):

    >>> _get_intervals(models.Box1D(width=3) | models.Shift(offset=2))
    (0.5,3.5)
    >>> _get_intervals(models.Box1D(width=3*u.m) | models.Shift(offset=2*u.m))
    (<Quantity 0.5 m>,<Quantity 3.5 m>)

    Multiplication (`*`) results in an intersection:

    >>> model1 = models.Box1D(x_0=-0.5, width=3)
    >>> model2 = models.Box1D(x_0=0.5, width=3)
    >>> _get_intervals(model1)
    (-2.0,1.0)
    >>> _get_intervals(model2)
    (-1.0,2.0)
    >>> _get_intervals(model1 * model2)
    (-1.0,1.0)
    >>> model1 = models.Box1D(x_0=-0.5*u.m, width=3*u.m)
    >>> model2 = models.Box1D(x_0=0.5*u.m, width=3*u.m)
    >>> _get_intervals(model1 * model2)
    (<Quantity -1. m>,<Quantity 1. m>)

    Addition (`+`) and subtraction (`-`) result in a union, but holes are kept.

    >>> model1 = models.Box1D(x_0=-0.5, width=3)
    >>> model2 = models.Box1D(x_0=0.5, width=3)
    >>> _get_intervals(model1 + model2)
    (-2.0,-1.0) | (-1.0,2.0)
    >>> _get_intervals(model1 - model2)
    (-2.0,-1.0) | (-1.0,2.0)
    >>> model1 = models.Box1D(x_0=-0.5*u.m, width=3*u.m)
    >>> model2 = models.Box1D(x_0=0.5*u.m, width=3*u.m)
    >>> _get_intervals(model1 + model2)
    (<Quantity -2. m>,<Quantity -1. m>) | (<Quantity -1. m>,<Quantity 2. m>)
    >>> _get_intervals(model1 - model2)
    (<Quantity -2. m>,<Quantity -1. m>) | (<Quantity -1. m>,<Quantity 2. m>)
    """
    if isinstance(model, CompoundModel):
        if model.op == "|":
            return _get_intervals(model.left).apply(_map_interval(model.right))
        elif model.op == "*":
            return _get_intervals(model.left) & _get_intervals(model.right)
        elif model.op in {"+", "-", "/"}:
            lhs = portion.IntervalDict({_get_intervals(model.left): "lhs"})
            rhs = portion.IntervalDict({_get_intervals(model.right): "rhs"})
            union = lhs | rhs
            return portion.Interval(
                *(intervals.apply(_to_open_interval) for intervals in union)
            )
        else:
            raise NotImplementedError(f"operation {model.op} not supported")
    else:
        try:
            bbox = model.bounding_box
        except NotImplementedError:
            unit = _get_1d_units_for_dict(model.input_units)
            return portion.open(-np.inf * unit, np.inf * unit)
        else:
            ((lo, hi),) = bbox
            return portion.open(_unwrap_scalar(lo), _unwrap_scalar(hi))


def integrate(
    model: Model, *, quick_and_dirty_npts: Optional[int] = None, **kwargs
) -> Union[Quantity, float]:
    """Integrate a 1D model using adaptive trapezoidal quadrature.

    Parameters
    ----------
    model : :class:`astropy.modeling.Model`
        The model to integrate.
    quick_and_dirty_npts : int, optional
        Number of sample points. If provided, disable adaptive quadrature and
        instead use fixed-order quadrature with the specified number of
        regularly spaced sample points.
    **kwargs : dict
        Additional keyword arguments passed to
        :meth:`scipy.integrate.quad_vec`.

    Returns
    -------
    integral : float, :class:`astropy.units.Quantity`
        The integral over the bounding box of the model's inputs.

    Notes
    -----
    When integrating compound models, the domain of integration is subdivided
    using the bounding box endpoints of all of the submodels. The integrator
    minimizes the error by tracking the error contributions from all of the
    subdivisions. The integrator can be slow; passing lower tolerances can
    speed up the calculation with a loss of accuracy. For a faster version
    without convergence checking, try the `quick_and_dirty_npts` option.

    Examples
    --------

    You can integrate dimensionless models:

    >>> from astropy.modeling import models
    >>> from astropy import units as u
    >>> from m4opt.models import integrate
    >>> import numpy as np
    >>> model = models.Box1D(width=3)
    >>> integrate(model)
    3.0
    >>> integrate(model, quick_and_dirty_npts=10000)
    3.0
    >>> model = models.Gaussian1D() * models.Const1D(1 / np.sqrt(2 * np.pi))
    >>> integrate(model, epsrel=1e-7)
    0.9999999648585338
    >>> integrate(model, quick_and_dirty_npts=10000)
    0.9999999620207557

    Or models with dimensions:

    >>> model = models.Lorentz1D(x_0=1 * u.micron, fwhm=0.1 * u.micron,
    ...                          amplitude=1 * u.erg / u.micron)
    >>> integrate(model, epsrel=1e-7)
    <Quantity 0.1550799 erg>
    >>> integrate(model, quick_and_dirty_npts=10000)
    <Quantity 0.1550799 erg>

    """
    x_unit = _get_1d_units_for_dict(model.input_units)

    intervals = _get_intervals(model)
    a, *points, b = np.unique(
        Quantity(
            list(chain.from_iterable((i.lower, i.upper) for i in intervals)), x_unit
        ).value.ravel()
    )

    # FIXME: Cannot rely on model.return_units because it is not set for all
    # models (e.g. Lorentz1D).
    y_unit = getattr(model(a * x_unit), "unit", None) or 1

    def func(x):
        result = model(x * x_unit)
        if hasattr(result, "to_value"):
            result = result.to_value()
        return result

    if quick_and_dirty_npts is not None:
        x = np.linspace(a, b, quick_and_dirty_npts)
        yint = trapezoid(func(x), x)
        if np.isscalar(yint):
            yint = _unwrap_scalar(yint.item())
    else:
        yint, _ = quad_vec(func, a, b, points=points, quadrature="trapezoid", **kwargs)

    return yint * x_unit * y_unit
