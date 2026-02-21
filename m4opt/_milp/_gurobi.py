"""Gurobi backend for MILP solver."""

from gzip import GzipFile
from io import BufferedWriter
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import gurobipy as gp
import numpy as np
from astropy import units as u
from gurobipy import GRB

from ._base import (
    ProgressData,
    ProgressDataRecorder,
    SolveDetails,
    VariableArray,
    add_var_array_method,
)

__all__ = ("GurobiModel", "GurobiSolveSolution")


def _unwrap(obj):
    """Unwrap a proxy object to get the underlying gurobipy object."""
    if isinstance(obj, GurobiVarProxy):
        return obj._var
    if isinstance(obj, GurobiExprProxy):
        return obj._expr
    return obj


def _unwrap_all(args):
    """Unwrap all proxy objects in an iterable."""
    return [_unwrap(a) for a in args]


class GurobiVarProxy:
    """Thin wrapper around gurobipy.Var for VariableArray operator compatibility.

    The main purpose is to make ``__eq__`` return an object that supports
    ``__rshift__`` for indicator constraint syntax: ``(x == 1) >> (y >= 0)``.
    """

    __slots__ = ("_var", "_model")

    def __init__(self, var, model):
        self._var = var
        self._model = model

    def __add__(self, other):
        return GurobiExprProxy(self._var + _unwrap(other), self._model)

    def __radd__(self, other):
        return GurobiExprProxy(other + self._var, self._model)

    def __sub__(self, other):
        return GurobiExprProxy(self._var - _unwrap(other), self._model)

    def __rsub__(self, other):
        return GurobiExprProxy(other - self._var, self._model)

    def __mul__(self, other):
        return GurobiExprProxy(self._var * _unwrap(other), self._model)

    def __rmul__(self, other):
        return GurobiExprProxy(_unwrap(other) * self._var, self._model)

    def __neg__(self):
        return GurobiExprProxy(-self._var, self._model)

    def __eq__(self, other):
        return GurobiEqualityProxy(self._var, _unwrap(other), self._model)

    def __ge__(self, other):
        return self._var >= _unwrap(other)

    def __le__(self, other):
        return self._var <= _unwrap(other)

    def __repr__(self):
        return repr(self._var)

    def __str__(self):
        return str(self._var)

    def __hash__(self):
        return hash(self._var)


class GurobiExprProxy:
    """Wrapper around gurobipy linear expression for operator compatibility."""

    __slots__ = ("_expr", "_model")

    def __init__(self, expr, model):
        self._expr = expr
        self._model = model

    def __add__(self, other):
        return GurobiExprProxy(self._expr + _unwrap(other), self._model)

    def __radd__(self, other):
        return GurobiExprProxy(_unwrap(other) + self._expr, self._model)

    def __sub__(self, other):
        return GurobiExprProxy(self._expr - _unwrap(other), self._model)

    def __rsub__(self, other):
        return GurobiExprProxy(_unwrap(other) - self._expr, self._model)

    def __mul__(self, other):
        return GurobiExprProxy(self._expr * _unwrap(other), self._model)

    def __rmul__(self, other):
        return GurobiExprProxy(_unwrap(other) * self._expr, self._model)

    def __neg__(self):
        return GurobiExprProxy(-self._expr, self._model)

    def __eq__(self, other):
        return GurobiEqualityProxy(self._expr, _unwrap(other), self._model)

    def __ge__(self, other):
        return self._expr >= _unwrap(other)

    def __le__(self, other):
        return self._expr <= _unwrap(other)


class GurobiEqualityProxy:
    """Result of ``var == val``. Supports ``__rshift__`` for indicator constraints."""

    __slots__ = ("_lhs", "_rhs", "_model")

    def __init__(self, lhs, rhs, model):
        self._lhs = lhs
        self._rhs = rhs
        self._model = model

    def __rshift__(self, constr):
        return GurobiIndicatorProxy(self._lhs, self._rhs, constr, self._model)

    def __bool__(self):
        raise TypeError(
            "Constraint truth value is ambiguous. "
            "Use model.add_constraints_() or model.add_indicator_constraints_()."
        )


class GurobiIndicatorProxy:
    """Represents an indicator constraint: ``(var == val) >> constr``."""

    __slots__ = ("_binvar", "_binval", "_constr", "_model")

    def __init__(self, binvar, binval, constr, model):
        self._binvar = binvar
        self._binval = int(binval)
        self._constr = constr
        self._model = model


class GurobiModel:
    """MILP model using Gurobi as the backend solver."""

    def __init__(
        self,
        timelimit: u.Quantity[u.physical.time] = 1e75 * u.s,
        jobs: int = 0,
        memory: u.Quantity[u.physical.data_quantity] = np.inf * u.byte,
        lowercutoff: float | None = None,
        verbose=True,
    ):
        """Initialize a model with default Gurobi parameters for M4OPT.

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
        """
        self._grb = gp.Model()

        # Vectorize abs
        self.abs = np.vectorize(self._abs_scalar)

        # Configure parameters
        self._grb.Params.LogToConsole = 1 if verbose else 0
        self._grb.Params.Threads = jobs

        timelimit_s = timelimit.to_value(u.s)
        if timelimit_s < 1e75:
            self._grb.Params.TimeLimit = timelimit_s
            # Emphasize finding good feasible solutions over proving optimality
            self._grb.Params.MIPFocus = 1

        if np.isfinite(memory):
            # NodefileStart: memory threshold (in GB) before writing nodes to disk
            self._grb.Params.NodefileStart = memory.to_value(u.GiB)

        if lowercutoff is not None:
            self._grb.Params.Cutoff = lowercutoff

        self._progress_recorder = None
        self._aux_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Don't dispose: solution values need the model to remain accessible.
        # Python's garbage collector handles cleanup when the model goes out of scope.
        pass

    def _create_var_list(self, tp, size, lb, ub):
        vtype_map = {
            "binary": GRB.BINARY,
            "continuous": GRB.CONTINUOUS,
            "integer": GRB.INTEGER,
            "semicontinuous": GRB.SEMICONT,
            "semiinteger": GRB.SEMIINT,
        }
        vtype = vtype_map[tp]
        vars = []
        for i in range(size):
            kwargs = {"vtype": vtype}
            if lb is not None:
                kwargs["lb"] = lb[i] if isinstance(lb, np.ndarray) else lb
            if ub is not None:
                kwargs["ub"] = ub[i] if isinstance(ub, np.ndarray) else ub
            vars.append(self._grb.addVar(**kwargs))
        self._grb.update()
        return [GurobiVarProxy(v, self) for v in vars]

    def _new_aux_var(self, lb=-GRB.INFINITY, ub=GRB.INFINITY):
        """Create a new auxiliary variable."""
        self._aux_count += 1
        v = self._grb.addVar(lb=lb, ub=ub, name=f"_aux{self._aux_count}")
        return v

    def _abs_scalar(self, expr):
        """Compute abs(expr), creating auxiliary variables and constraints."""
        expr = _unwrap(expr)
        if isinstance(expr, (int, float)):
            return abs(expr)
        # If expr is a LinExpr (not a Var), create a helper var first
        if not isinstance(expr, gp.Var):
            helper = self._new_aux_var()
            self._grb.addConstr(helper == expr)
            expr = helper
        aux = self._new_aux_var(lb=0)
        self._grb.addGenConstrAbs(aux, expr)
        return GurobiVarProxy(aux, self)

    def binary_var(self, name=None):
        """Create a single binary decision variable."""
        return self.binary_vars()

    def continuous_var(self, name=None, lb=None, ub=None):
        """Create a single continuous decision variable."""
        return self.continuous_vars(lb=lb, ub=ub)

    def integer_var(self, name=None, lb=None, ub=None):
        """Create a single integer decision variable."""
        return self.integer_vars(lb=lb, ub=ub)

    def add_constraint_(self, ct, name=None):
        """Add a single constraint to the model."""
        ct = _unwrap(ct)
        if isinstance(ct, GurobiEqualityProxy):
            self._grb.addConstr(ct._lhs == ct._rhs)
        else:
            self._grb.addConstr(ct)

    def add_constraints_(self, cts, names=None):
        """Add any number of constraints to the model."""
        if isinstance(cts, np.ndarray):
            cts = cts.ravel()
        for ct in cts:
            self.add_constraint_(ct)

    def add_indicators(self, binary_vars, cts, true_values=1, names=None):
        """Add any number of indicator constraints to the model."""
        binary_vars = np.asarray(binary_vars).ravel()
        cts = np.asarray(cts).ravel()
        true_values = np.broadcast_to(true_values, len(binary_vars)).ravel()
        for bvar, ct, tv in zip(binary_vars, cts, true_values):
            bvar = _unwrap(bvar)
            self._grb.addGenConstrIndicator(bvar, int(tv), ct)

    def add_indicator_constraints(self, indcts):
        """Add indicator constraints from ``(var == val) >> constr`` expressions."""
        indcts = np.asarray(indcts)
        for ic in indcts.ravel():
            if isinstance(ic, GurobiIndicatorProxy):
                self._grb.addGenConstrIndicator(
                    ic._binvar, ic._binval, ic._constr
                )
            else:
                raise TypeError(f"Expected GurobiIndicatorProxy, got {type(ic)}")

    def add_indicator_constraints_(self, indcts):
        """Same as add_indicator_constraints (batch version)."""
        self.add_indicator_constraints(indcts)

    def add_user_cut_constraint(self, ct):
        """Add a user cut constraint (added as a regular constraint in Gurobi)."""
        self.add_constraint_(ct)

    def maximize(self, expr):
        """Set the objective to maximize the given expression."""
        self._grb.setObjective(_unwrap(expr), GRB.MAXIMIZE)

    def minimize(self, expr):
        """Set the objective to minimize the given expression."""
        self._grb.setObjective(_unwrap(expr), GRB.MINIMIZE)

    def _ensure_var(self, expr):
        """Ensure an expression is a Var (create auxiliary variable if needed)."""
        expr = _unwrap(expr)
        if isinstance(expr, gp.Var):
            return expr
        helper = self._new_aux_var()
        self._grb.addConstr(helper == expr)
        return helper

    def min(self, *args):
        """Return the minimum of the given variables/expressions."""
        vars = [self._ensure_var(a) for a in args]
        aux = self._new_aux_var()
        self._grb.addGenConstrMin(aux, vars)
        return np.asarray(GurobiVarProxy(aux, self)).view(VariableArray)

    def max(self, *args):
        """Return the maximum of the given variables/expressions."""
        vars = [self._ensure_var(a) for a in args]
        aux = self._new_aux_var()
        self._grb.addGenConstrMax(aux, vars)
        return np.asarray(GurobiVarProxy(aux, self)).view(VariableArray)

    def sum(self, args):
        """Sum the given expressions."""
        return gp.quicksum(_unwrap(a) for a in args)

    def sum_vars_all_different(self, vars):
        """Sum variables (equivalent to quicksum for Gurobi)."""
        if isinstance(vars, np.ndarray):
            vars = vars.ravel()
        return gp.quicksum(_unwrap(v) for v in vars)

    def scal_prod_vars_all_different(self, vars, coeffs):
        """Compute the scalar product of variables and coefficients."""
        if isinstance(vars, np.ndarray):
            vars = vars.ravel()
        if isinstance(coeffs, np.ndarray):
            coeffs = coeffs.ravel()
        return gp.LinExpr(list(coeffs), [_unwrap(v) for v in vars])

    def piecewise(self, preslope, breakpoints, postslope):
        """Return a callable that creates a piecewise linear constraint.

        Parameters
        ----------
        preslope
            Slope before the first breakpoint.
        breakpoints
            Sequence of (x, y) breakpoint tuples.
        postslope
            Slope after the last breakpoint.
        """

        def apply(var):
            var = _unwrap(var)
            xpts = [bp[0] for bp in breakpoints]
            ypts = [bp[1] for bp in breakpoints]

            # Extend with pre/post slopes if needed
            if preslope != 0 and len(xpts) > 0:
                x_pre = xpts[0] - 1.0
                y_pre = ypts[0] - preslope
                xpts.insert(0, x_pre)
                ypts.insert(0, y_pre)
            if postslope != 0 and len(xpts) > 0:
                x_post = xpts[-1] + 1.0
                y_post = ypts[-1] + postslope
                xpts.append(x_post)
                ypts.append(y_post)

            aux = self._new_aux_var()
            self._grb.addGenConstrPWL(var, aux, xpts, ypts)
            return GurobiVarProxy(aux, self)

        return apply

    def add_progress_listener(self, recorder):
        """Register a progress data recorder."""
        if isinstance(recorder, ProgressDataRecorder):
            self._progress_recorder = recorder
        else:
            self._progress_recorder = recorder

    def solve(self, **kwargs):
        """Solve the model.

        Returns
        -------
        GurobiSolveSolution or None
            The solution if one was found, otherwise None.
        """
        # Build callback for progress recording if needed
        if self._progress_recorder is not None:

            def callback(model, where):
                if where == GRB.Callback.MIP:
                    try:
                        obj_bst = model.cbGet(GRB.Callback.MIP_OBJBST)
                        obj_bnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                        sol_cnt = int(model.cbGet(GRB.Callback.MIP_SOLCNT))
                        node_cnt = int(model.cbGet(GRB.Callback.MIP_NODCNT))
                        node_left = int(model.cbGet(GRB.Callback.MIP_NODLFT))
                        iter_cnt = int(model.cbGet(GRB.Callback.MIP_ITRCNT))
                        runtime = model.cbGet(GRB.Callback.RUNTIME)

                        gap = abs(obj_bst - obj_bnd) / max(abs(obj_bst), 1e-10)

                        self._progress_recorder.record(
                            ProgressData(
                                current_nb_iterations=iter_cnt,
                                has_incumbent=sol_cnt > 0,
                                current_objective=obj_bst,
                                best_bound=obj_bnd,
                                current_mip_gap=gap,
                                current_nb_nodes=node_cnt,
                                remaining_nb_nodes=node_left,
                                current_nb_solutions=sol_cnt,
                                time=runtime,
                                det_time=runtime,
                            )
                        )
                    except gp.GurobiError:
                        pass

            self._grb.optimize(callback)
        else:
            self._grb.optimize()

        # Build solve details
        status = self._grb.Status
        status_map = {
            GRB.OPTIMAL: "optimal",
            GRB.INFEASIBLE: "infeasible",
            GRB.INF_OR_UNBD: "infeasible or unbounded",
            GRB.UNBOUNDED: "unbounded",
            GRB.CUTOFF: "aborted, lower cutoff reached",
            GRB.ITERATION_LIMIT: "iteration limit",
            GRB.NODE_LIMIT: "node limit",
            GRB.TIME_LIMIT: "time limit",
            GRB.SOLUTION_LIMIT: "solution limit",
            GRB.INTERRUPTED: "interrupted",
            GRB.NUMERIC: "numeric",
            GRB.SUBOPTIMAL: "suboptimal",
            GRB.USER_OBJ_LIMIT: "user objective limit",
        }
        status_str = status_map.get(status, f"unknown ({status})")

        self._solve_details = SolveDetails(
            status=status_str,
            time=self._grb.Runtime,
        )

        # Return solution if available
        if self._grb.SolCount > 0:
            return GurobiSolveSolution(self._grb, self._solve_details)
        return None

    @property
    def best_bound(self):
        """Get the best bound from the last solve."""
        try:
            return self._grb.ObjBound
        except gp.GurobiError:
            return float("inf")

    @property
    def solve_details(self):
        """Get solve details from the last solve."""
        if hasattr(self, "_solve_details"):
            return self._solve_details
        return SolveDetails(status="not solved", time=0.0)

    @property
    def objective_value(self):
        """Get the objective value from the last solve."""
        return self._grb.ObjVal

    def to_stream(self, out_file: BufferedWriter):
        """Write the model to a stream.

        The filename should end in `.lp`, `.mps`, `.lp.gz`, or `.mps.gz`.
        """
        valid_formats = {"lp", "mps"}
        out_filename = out_file.name
        out_path = Path(out_filename)
        suffixes = Path(out_path).suffixes

        if (
            len(suffixes) == 0
            or (fmt := suffixes[0].lstrip(".").lower()) not in valid_formats
        ):
            valid_extensions = [f".{f}" for f in valid_formats]
            valid_extensions = [
                *valid_extensions,
                *(f"{ext}.gz" for ext in valid_extensions),
            ]
            raise ValueError(
                f'Invalid model filename "{out_filename}". The extension must '
                f'be one of the following: {" ".join(valid_extensions)}'
            )

        should_gzip = suffixes[-1].lower() == ".gz"

        if should_gzip:
            with NamedTemporaryFile(suffix=f".{fmt}") as temp_file:
                self._grb.write(temp_file.name)
                temp_file.seek(0)
                with GzipFile(
                    f"{out_path.name}{suffixes[0]}", "wb", fileobj=out_file
                ) as zip_file:
                    copyfileobj(temp_file, zip_file)
        else:
            self._grb.write(out_filename)


class GurobiSolveSolution:
    """Solution from a Gurobi solve."""

    def __init__(self, grb_model, solve_details):
        self._grb = grb_model
        self.solve_details = solve_details

    def get_values(self, var_seq):
        """Get solution values for multidimensional arrays of variables."""
        var_seq = np.asarray(var_seq)
        values = np.array([_unwrap(v).X for v in var_seq.ravel()])
        return values.reshape(var_seq.shape)

    def get_objective_value(self):
        """Get the objective value of this solution."""
        return self._grb.ObjVal


for _tp in ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]:
    add_var_array_method(GurobiModel, _tp)
del _tp
