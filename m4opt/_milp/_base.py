"""Backend-agnostic base classes and utilities for MILP solvers."""

import operator
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from ..utils.console import status

__all__ = ("ProgressData", "ProgressDataRecorder", "SolveDetails", "VariableArray")


class VariableArray(np.ndarray):
    """Subclass numpy.ndarray to support vectorized comparison operators."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc not in ufunc_map:
            return (
                super()
                .__array_ufunc__(
                    ufunc,
                    method,
                    *(
                        input.view(np.ndarray)
                        if isinstance(input, VariableArray)
                        else input
                        for input in inputs
                    ),
                    **kwargs,
                )
                .view(self.__class__)
            )
        elif method != "__call__":
            return NotImplemented
        else:
            return ufunc_map[ufunc](
                *(input.view(self.__class__) for input in inputs)
            ).view(self.__class__)


def _make_attr(op):
    setattr(
        VariableArray,
        op,
        lambda self, rhs: getattr(super(VariableArray, self), op)(
            np.asarray(rhs, VariableArray)
        ),
    )
    return np.vectorize(getattr(operator, op), signature="(),()->()")


ufunc_map = {
    numpyfunc: _make_attr(op)
    for numpyfunc, op in [
        [np.add, "__add__"],
        [np.equal, "__eq__"],
        [np.greater_equal, "__ge__"],
        [np.less_equal, "__le__"],
        [np.right_shift, "__rshift__"],
        [np.subtract, "__sub__"],
    ]
}

del _make_attr


@dataclass
class SolveDetails:
    """Backend-agnostic solve details."""

    status: str
    time: float  # seconds


# ProgressData namedtuple compatible with docplex's ProgressData fields.
ProgressData = namedtuple(
    "ProgressData",
    [
        "current_nb_iterations",
        "has_incumbent",
        "current_objective",
        "best_bound",
        "current_mip_gap",
        "current_nb_nodes",
        "remaining_nb_nodes",
        "current_nb_solutions",
        "time",
        "det_time",
    ],
)


class ProgressDataRecorder:
    """Backend-agnostic progress data recorder."""

    def __init__(self):
        self.recorded = []

    def record(self, data):
        self.recorded.append(data)


def add_var_array_method(cls, tp):
    """Dynamically add a ``{tp}_vars`` method to a Model class."""

    def func(self, shape=(), lb=None, ub=None):
        size = np.prod(shape, dtype=int)
        if lb is not None:
            lb = np.broadcast_to(lb, shape).ravel()
            if lb.size == 1:
                lb = lb.item()
        if ub is not None:
            ub = np.broadcast_to(ub, shape).ravel()
            if ub.size == 1:
                ub = ub.item()
        with status(f"adding {size} {tp} variables"):
            vars = np.reshape(self._create_var_list(tp, size, lb, ub), shape).view(
                VariableArray
            )
            if vars.ndim == 0:
                vars = vars.item()
            return vars

    func.__doc__ = f"""Create an arbitrary N-dimensional array of {tp} decision variables.

    Parameters
    ----------
    shape
        The desired shape of the array.
    lb
        Lower bound for the variables.
    ub
        Upper bound for the variables.

    Examples
    --------
    >>> from m4opt.milp import Model
    >>> import numpy as np
    >>> model = Model()
    >>> x = model.{tp}_vars()
    \u2713 adding 1 {tp} variables 0:00:00
    >>> y = model.{tp}_vars(3, lb=1, ub=1)
    \u2713 adding 3 {tp} variables 0:00:00
    >>> z = model.{tp}_vars((3, 4), lb=np.ones((3, 4)), ub=np.ones((3, 4)))
    \u2713 adding 12 {tp} variables 0:00:00
    """

    setattr(cls, f"{tp}_vars", func)
