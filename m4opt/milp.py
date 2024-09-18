"""Mixed integer linear programs (MILP)"""

import operator

import numpy as np
from astropy import units as u
from docplex.mp.model import Model as _Model

from .utils.console import status
from .utils.numpy import atmost_1d

__all__ = ("Model",)


class Model(_Model):
    """Convenience class to add Numpy variable arrays to a CPLEX model."""

    def __init__(
        self, timelimit: u.Quantity[u.physical.time] = 1e75 * u.s, jobs: int = 0
    ):
        """Initialize a model with default `CPLEX parameters`_ for M4OPT.

        Parameters
        ----------
        timelimit : astropy.units.Quantity
            Maximum solver run time. Default: run until the solver terminates
            naturally due to finding the optimum or proving the problem to be
            infeasible.
        jobs : int
            Number of threads. If 0, then automatically configure the number of
            threads based on the number of CPUs present.

        Notes
        -----
        - If a time limit is provided, then set the `solver emphasis`_ to
          finding high-quality feasible solutions rather than proving bounds on
          the objective value.

        - Disable the `solution pool`_ to save memory because we only care
          about finding one high-quality solution and not multiple feasible
          solutions.

        - Enable `opportunistic parallelism`_ for faster solutions on many-core
          systems at the expense of results that may vary from run to run.

        .. _`CPLEX parameters`: https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-list-parameters
        .. _`solver emphasis`: https://www.ibm.com/docs/en/icos/22.1.1?topic=parameters-mip-emphasis-switch
        .. _`solution pool`: https://www.ibm.com/docs/en/icos/22.1.1?topic=parameters-maximum-number-solutions-kept-in-solution-pool
        .. _`opportunistic parallelism`: https://www.ibm.com/docs/en/icos/22.1.1?topic=parameters-parallel-mode-switch
        """
        super().__init__()

        self.abs = np.vectorize(self.abs)

        self.context.solver.log_output = True
        self.context.cplex_parameters.threads = jobs

        # Disable deterministic parallelism. We're OK with results being
        # slightly dependent upon timing.
        self.context.cplex_parameters.parallel = (
            self.cplex.parameters.parallel.values.opportunistic
        )

        # Disable the solution pool. We are not examining multiple solutions,
        # and the solution pool can grow to take up a lot of memory.
        self.context.cplex_parameters.mip.pool.capacity = 0

        timelimit_s = timelimit.to_value(u.s)
        if timelimit_s < 1e75:
            self.context.cplex_parameters.timelimit = timelimit_s
            # Since we have a time limit,
            # emphasize finding good feasible solutions over proving optimality.
            self.context.cplex_parameters.emphasis.mip = (
                self.cplex.parameters.emphasis.mip.values.feasibility
            )

    def add_constraints_(self, cts, names=None):
        """Add any number of constraints to the model.

        Examples
        --------
        This method adds support for arrays of constraints to
        :meth:`docplex.mp.model.Model.add_constraints_`:

        >>> from m4opt.milp import Model
        >>> import numpy as np
        >>> m = Model()
        >>> x = m.continuous_vars((3, 4))
        ✓ adding 12 continuous variables 0:00:00
        >>> xmax = np.random.normal(size=x.shape)
        >>> m.add_constraints_(x >= xmax)
        """
        return super().add_constraints_(atmost_1d(cts), names)

    def add_indicators(self, binary_vars, cts, true_values=1, names=None):
        """Add any number of indicator constraints to the model.

        Examples
        --------
        This method adds support for arrays of constraints to
        :meth:`docplex.mp.model.Model.add_indicators`:

        >>> from m4opt.milp import Model
        >>> import numpy as np
        >>> m = Model()
        >>> x = m.continuous_vars((3, 4))
        ✓ adding 12 continuous variables 0:00:00
        >>> y = m.binary_vars((3, 4))
        ✓ adding 12 binary variables 0:00:00
        >>> xmax = np.random.normal(size=x.shape)
        >>> _ = m.add_indicators(y, x >= xmax)
        """
        return super().add_indicators(
            atmost_1d(binary_vars), atmost_1d(cts), atmost_1d(true_values), names
        )


ufunc_map = {
    np.less_equal: np.vectorize(operator.le, signature="(),()->()"),
    np.greater_equal: np.vectorize(operator.ge, signature="(),()->()"),
    np.equal: np.vectorize(operator.eq, signature="(),()->()"),
}


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
            return ufunc_map[ufunc](*inputs)


def add_var_array_method(cls, tp):
    def func(self, shape=(), lb=None, ub=None):
        size = np.prod(shape, dtype=int)
        if lb is not None:
            lb = np.ravel(lb)
        if ub is not None:
            ub = np.ravel(ub)
        with status(f"adding {size} {tp} variables"):
            vartype = getattr(self, f"{tp}_vartype")
            vars = np.reshape(self.var_list(size, vartype, lb, ub), shape).view(
                VariableArray
            )
            if vars.ndim == 0:
                vars = vars.item()
            return vars

    func.__doc__ = f"""Create an arbitary N-dimensional array of {tp} decision variables.

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

    Examples
    --------
    >>> from m4opt.milp import Model
    >>> model = Model()
    >>> x = model.{tp}_vars()
    ✓ adding 1 {tp} variables 0:00:00
    >>> y = model.{tp}_vars(3)
    ✓ adding 3 {tp} variables 0:00:00
    >>> z = model.{tp}_vars((3, 4))
    ✓ adding 12 {tp} variables 0:00:00
    >>> print(x)
    x1
    >>> print(y)
    [x2 x3 x4]
    >>> print(z)
    [[x5 x6 x7 x8]
     [x9 x10 x11 x12]
     [x13 x14 x15 x16]]
    """

    setattr(cls, f"{tp}_vars", func)


for tp in ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]:
    add_var_array_method(Model, tp)
