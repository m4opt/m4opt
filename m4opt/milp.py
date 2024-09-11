"""Mixed integer linear programs (MILP)"""

import logging

import numpy as np
from docplex.mp.model import Model as _Model

__all__ = ("Model",)

log = logging.getLogger(__name__)


class Model(_Model):
    """Convenience class to add Numpy variable arrays to a CPLEX model."""


def add_var_array_method(cls, tp):
    def func(self, shape=(), *args, **kwargs):
        vartype = getattr(self, f"{tp}_vartype")
        size = np.prod(shape, dtype=int)
        vars = np.reshape(self.var_list(size, vartype, *args, **kwargs), shape)
        if vars.ndim == 0:
            vars = vars.item()
        log.debug("Added %d %s variables", size, tp)
        return vars

    func.__doc__ = f"""Create a Numpy array of {tp} decision variables.

    Parameters
    ----------
    shape : int, tuple
        The desired shape of the array.
    args, kwargs
        Additional arguments passed to
        :meth:`~docplex.mp.model.Model.{tp}_var_list`, such as lower and upper
        bounds.

    Returns
    -------
    numpy.ndarray
    """

    setattr(cls, f"{tp}_vars", func)


for tp in ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]:
    add_var_array_method(Model, tp)
