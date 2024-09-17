"""Mixed integer linear programs (MILP)"""

import numpy as np
from astropy import units as u
from docplex.mp.model import Model as _Model

from .utils.console import status

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


def add_var_array_method(cls, tp):
    def func(self, shape=(), lb=None, ub=None):
        size = np.prod(shape, dtype=int)
        if lb is not None:
            lb = np.ravel(lb)
        if ub is not None:
            ub = np.ravel(ub)
        with status(f"adding {size} {tp} variables"):
            vartype = getattr(self, f"{tp}_vartype")
            vars = np.reshape(self.var_list(size, vartype, lb, ub), shape)
            if vars.ndim == 0:
                vars = vars.item()
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
