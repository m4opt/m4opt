"""CPLEX/DOcplex backend for MILP solver."""

from collections.abc import Callable
from gzip import GzipFile
from io import BufferedWriter
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, gettempdir
from unittest.mock import patch

import cplex
import numpy as np
from astropy import units as u
from docplex.mp.model import Model as _Model
from docplex.mp.progress import ProgressDataRecorder as _ProgressDataRecorder
from docplex.mp.solution import SolveSolution as _SolveSolution
from numpy import typing as npt

from ..utils.numpy import atmost_1d
from ._base import ProgressData, ProgressDataRecorder, VariableArray, add_var_array_method

__all__ = ("CplexModel", "CplexSolveSolution")


class LowerCutoffCallback:
    def __init__(self, cutoff: float):
        self._cutoff = cutoff
        self._reached = False

    def invoke(self, context: cplex.callbacks.Context):
        best_bound = context.get_double_info(cplex.callbacks.CallbackInfo.best_bound)
        # Note that CPLEX indicates that it has not yet found a best bound by
        # setting this to a large negative value.
        if -cplex.infinity < best_bound <= self._cutoff:
            print(
                f"giving up because best bound ({best_bound}) <= cutoff ({self._cutoff})"
            )
            self._reached = True
            context.abort()


class _ProgressBridge(_ProgressDataRecorder):
    """Bridge docplex ProgressDataRecorder to our backend-agnostic recorder."""

    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder

    def notify_progress(self, progress_data):
        super().notify_progress(progress_data)
        self._recorder.record(
            ProgressData(
                current_nb_iterations=progress_data.current_nb_iterations,
                has_incumbent=progress_data.has_incumbent,
                current_objective=progress_data.current_objective,
                best_bound=progress_data.best_bound,
                current_mip_gap=progress_data.current_mip_gap,
                current_nb_nodes=progress_data.current_nb_nodes,
                remaining_nb_nodes=progress_data.remaining_nb_nodes,
                current_nb_solutions=progress_data.current_nb_solutions,
                time=progress_data.time,
                det_time=progress_data.det_time,
            )
        )


class CplexModel(_Model):
    """Convenience class to add Numpy variable arrays to a CPLEX model."""

    def __init__(
        self,
        timelimit: u.Quantity[u.physical.time] = 1e75 * u.s,
        jobs: int = 0,
        memory: u.Quantity[u.physical.data_quantity] = np.inf * u.byte,
        lowercutoff: float | None = None,
        verbose=True,
    ):
        """Initialize a model with default `CPLEX parameters`_ for M4OPT.

        Parameters
        ----------
        timelimit
            Maximum solver run time. Default: run until the solver terminates
            naturally due to finding the optimum or proving the problem to be
            infeasible.
        jobs
            Number of threads. If 0, then automatically configure the number of
            threads based on the number of CPUs present.
        memory
            Maximum memory usage before terminating the solver.
        lowercutoff
            Optional lower cutoff. Terminate the solver if the best bound drops
            below this value.
        verbose
            Display live solver progress.

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

        self.abs: Callable[[npt.ArrayLike], npt.ArrayLike] = np.vectorize(self.abs)

        self.context.solver.log_output = verbose
        self.context.cplex_parameters.threads = jobs
        self.context.cplex_parameters.workdir = gettempdir()

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

        if np.isfinite(memory):
            self.context.cplex_parameters.mip.strategy.file = 3
            self.context.cplex_parameters.workmem = memory.to_value(u.MiB)

        if lowercutoff is not None:
            # FIXME: Setting lowercutoff drastically hurts solution quality.
            # As a workaround, we install a callback instead.
            # See https://github.com/IBMDecisionOptimization/docplex/issues/20
            #
            # model.context.cplex_parameters.mip.tolerances.lowercutoff = cutoff
            self._lower_cutoff = LowerCutoffCallback(lowercutoff)
            self.get_cplex().set_callback(
                self._lower_cutoff,
                cplex.callbacks.Context.id.global_progress,
            )

    def _create_var_list(self, tp, size, lb, ub):
        vartype = getattr(self, f"{tp}_vartype")
        return self.var_list(size, vartype, lb, ub)

    # FIXME: remove once
    # https://github.com/IBMDecisionOptimization/docplex/issues/17 is fixed.
    @property
    def best_bound(self):
        """Get best bound for the last solve.

        Notes
        -----
        This is provided as a workaround for a bug in DOcplex where
        :obj:`docplex.mp.sdetails.SolveDetails` is not updated if CPLEX reaches
        its time limit without finding an integer solution. See
        https://github.com/IBMDecisionOptimization/docplex/issues/17.
        """
        return self.cplex.solution.MIP.get_best_objective()

    def add_constraints_(self, cts, names=None):
        """Add any number of constraints to the model.

        Examples
        --------
        This method adds support for arrays of constraints to
        :meth:`docplex.mp.model.Model.add_constraints_`:

        >>> from m4opt._milp._cplex import CplexModel as Model
        >>> import numpy as np
        >>> m = Model()
        >>> x = m.continuous_vars((3, 4))
        \u2713 adding 12 continuous variables 0:00:00
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

        >>> from m4opt._milp._cplex import CplexModel as Model
        >>> import numpy as np
        >>> m = Model()
        >>> x = m.continuous_vars((3, 4))
        \u2713 adding 12 continuous variables 0:00:00
        >>> y = m.binary_vars((3, 4))
        \u2713 adding 12 binary variables 0:00:00
        >>> xmax = np.random.normal(size=x.shape)
        >>> _ = m.add_indicators(y, x >= xmax)
        """
        return super().add_indicators(
            atmost_1d(binary_vars), atmost_1d(cts), atmost_1d(true_values), names
        )

    def add_indicator_constraints(self, indcts):
        return super().add_indicator_constraints(atmost_1d(indcts))

    def add_indicator_constraints_(self, indcts):
        return super().add_indicator_constraints_(atmost_1d(indcts))

    def add_progress_listener(self, recorder):
        if isinstance(recorder, ProgressDataRecorder):
            bridge = _ProgressBridge(recorder)
            super().add_progress_listener(bridge)
        else:
            super().add_progress_listener(recorder)

    def solve(self, **kwargs):
        with patch("docplex.mp.solution.SolveSolution", CplexSolveSolution):
            result = super().solve(**kwargs)
        if (
            (cutoff := getattr(self, "_lower_cutoff", None))
            and cutoff._reached
            and self.solve_details.status == "aborted"
        ):
            message = "aborted, lower cutoff reached"
            self._solve_details._solve_status = message
            if result is not None:
                result.solve_details._solve_status = message
        return result

    def min(self, *args):
        return np.asarray(super().min(*args)).view(VariableArray)

    def max(self, *args):
        return np.asarray(super().max(*args)).view(VariableArray)

    def to_stream(self, out_file: BufferedWriter):
        """Write the model to a stream.

        The filename should end in `.lp`, `.mps`, `.sav`, `.lp.gz`, `.mps.gz`,
        or `.sav.gz`.
        """
        valid_formats = {"lp", "mps", "sav"}
        out_filename = out_file.name
        out_path = Path(out_filename)
        suffixes = Path(out_path).suffixes

        if (
            len(suffixes) == 0
            or (format := suffixes[0].lstrip(".").lower()) not in valid_formats
        ):
            valid_extensions = [f".{fmt}" for fmt in valid_formats]
            valid_extensions = [
                *valid_extensions,
                *(f"{ext}.gz" for ext in valid_extensions),
            ]
            raise ValueError(
                f'Invalid model filename "{out_filename}". The extension must be one of the following: {" ".join(valid_extensions)}'
            )
        export_method = getattr(self, f"export_as_{format}")

        should_gzip = suffixes[-1].lower() == ".gz"

        if should_gzip:
            with NamedTemporaryFile(suffix=f".{format}") as temp_file:
                export_method(temp_file.name)
                with GzipFile(
                    f"{out_path.name}{suffixes[0]}", "wb", fileobj=out_file
                ) as zip_file:
                    copyfileobj(temp_file, zip_file)
        else:
            export_method(out_filename)


class CplexSolveSolution(_SolveSolution):
    def get_values(self, var_seq):
        """Get solution values for multidimensional arrays of variables.

        Examples
        --------
        >>> from m4opt._milp._cplex import CplexModel as Model
        >>> m = Model()
        >>> x = m.continuous_vars((3, 4), ub=42)
        \u2713 adding 12 continuous variables 0:00:00
        >>> m.maximize(m.sum(x.ravel()))
        >>> solution = m.solve()  # doctest: +ELLIPSIS
        Version identifier: ...
        >>> solution.get_values(x[0])
        array([42., 42., 42., 42.])
        >>> solution.get_values(x)
        array([[42., 42., 42., 42.],
               [42., 42., 42., 42.],
               [42., 42., 42., 42.]])
        """
        var_seq = np.asarray(var_seq)
        return np.asarray(super().get_values(var_seq.ravel())).reshape(var_seq.shape)


for _tp in ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]:
    add_var_array_method(CplexModel, _tp)
del _tp
