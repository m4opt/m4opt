"""SCIP backend for MILP solver."""

from gzip import GzipFile
from io import BufferedWriter
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile

import numpy as np
import pyscipopt as scip
from astropy import units as u
from pyscipopt import SCIP_EVENTTYPE, SCIP_PARAMEMPHASIS, SCIP_PARAMSETTING, Eventhdlr

from ..utils.console import status
from ._base import (
    _VARIABLE_TYPES,
    ProgressData,
    SolveDetails,
    VariableArray,
    add_var_array_method,
)

__all__ = ("SCIPModel", "SCIPSolveSolution")

_INFINITY = 1e20


def _unwrap(obj):
    """Return the underlying SCIP object of a proxy."""
    if isinstance(obj, SCIPVarProxy):
        return obj._var
    if isinstance(obj, SCIPExprProxy):
        return obj._expr
    return obj


class SCIPVarProxy:
    """Wrap a SCIP variable so that ``==`` can build an indicator constraint.

    ``var == value`` must return something that supports ``>>``, which the
    native equality does not.
    """

    __slots__ = ("_model", "_var")

    def __init__(self, var, model):
        self._var = var
        self._model = model

    def __add__(self, other):
        return SCIPExprProxy(self._var + _unwrap(other), self._model)

    def __radd__(self, other):
        return SCIPExprProxy(_unwrap(other) + self._var, self._model)

    def __sub__(self, other):
        return SCIPExprProxy(self._var - _unwrap(other), self._model)

    def __rsub__(self, other):
        return SCIPExprProxy(_unwrap(other) - self._var, self._model)

    def __mul__(self, other):
        return SCIPExprProxy(self._var * _unwrap(other), self._model)

    def __rmul__(self, other):
        return SCIPExprProxy(_unwrap(other) * self._var, self._model)

    def __neg__(self):
        return SCIPExprProxy(-self._var, self._model)

    def __eq__(self, other):
        return SCIPEqualityProxy(self._var, _unwrap(other), self._model)

    def __ge__(self, other):
        return self._var >= _unwrap(other)

    def __le__(self, other):
        return self._var <= _unwrap(other)

    @property
    def lb(self):
        return self._var.getLbOriginal()

    @property
    def ub(self):
        return self._var.getUbOriginal()

    def __repr__(self):
        return repr(self._var)

    def __str__(self):
        return str(self._var)

    def __hash__(self):
        return hash(self._var)


class SCIPExprProxy:
    """Wrap a SCIP expression to keep proxy semantics through arithmetic."""

    __slots__ = ("_expr", "_model")

    def __init__(self, expr, model):
        self._expr = expr
        self._model = model

    def __add__(self, other):
        return SCIPExprProxy(self._expr + _unwrap(other), self._model)

    def __radd__(self, other):
        return SCIPExprProxy(_unwrap(other) + self._expr, self._model)

    def __sub__(self, other):
        return SCIPExprProxy(self._expr - _unwrap(other), self._model)

    def __rsub__(self, other):
        return SCIPExprProxy(_unwrap(other) - self._expr, self._model)

    def __mul__(self, other):
        return SCIPExprProxy(self._expr * _unwrap(other), self._model)

    def __rmul__(self, other):
        return SCIPExprProxy(_unwrap(other) * self._expr, self._model)

    def __neg__(self):
        return SCIPExprProxy(-self._expr, self._model)

    def __eq__(self, other):
        return SCIPEqualityProxy(self._expr, _unwrap(other), self._model)

    def __ge__(self, other):
        return self._expr >= _unwrap(other)

    def __le__(self, other):
        return self._expr <= _unwrap(other)


class SCIPEqualityProxy:
    """Result of ``var == val``. Supports ``__rshift__`` for indicators."""

    __slots__ = ("_lhs", "_model", "_rhs")

    def __init__(self, lhs, rhs, model):
        self._lhs = lhs
        self._rhs = rhs
        self._model = model

    def __rshift__(self, constr):
        return SCIPIndicatorProxy(self._lhs, self._rhs, constr, self._model)

    def __bool__(self):
        raise TypeError(
            "Constraint truth value is ambiguous. "
            "Use model.add_constraints_() or model.add_indicator_constraints_()."
        )


class SCIPIndicatorProxy:
    """Result of ``(var == val) >> constraint``."""

    __slots__ = ("_binval", "_binvar", "_constr", "_model")

    def __init__(self, binvar, binval, constr, model):
        self._binvar = binvar
        self._binval = binval
        self._constr = constr
        self._model = model


class _ProgressEventHandler(Eventhdlr):
    """Record solver progress each time the incumbent improves."""

    def __init__(self, recorder):
        super().__init__()
        self._recorder = recorder
        self._count = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        m = self.model
        self._count += 1
        # An exception raised here aborts the solve, and not every event fires
        # in a stage where the statistics can be queried.
        try:
            objective = m.getPrimalbound()
            bound = m.getDualbound()
            runtime = m.getSolvingTime()
            self._recorder.record(
                ProgressData(
                    current_nb_iterations=int(m.getNLPIterations() or 0),
                    has_incumbent=True,
                    current_objective=objective,
                    best_bound=bound,
                    current_mip_gap=abs(objective - bound) / max(abs(objective), 1e-10),
                    current_nb_nodes=int(m.getNNodes()),
                    remaining_nb_nodes=int(m.getNNodesLeft()),
                    current_nb_solutions=self._count,
                    time=runtime,
                    det_time=runtime,
                )
            )
        except Exception:  # noqa: BLE001
            return


def _flatten_constraints(cts):
    """Normalize one constraint, or any container of them, to a flat sequence."""
    if isinstance(cts, (scip.scip.ExprCons, SCIPEqualityProxy)):
        return (cts,)
    if isinstance(cts, np.ndarray):
        return cts.ravel()
    return cts


class SCIPModel:
    """A MILP model backed by SCIP."""

    def __init__(
        self,
        timelimit=np.inf * u.s,
        jobs=0,
        memory=np.inf * u.MiB,
        lowercutoff=None,
        verbose=True,
    ):
        self._scip = scip.Model()
        self._progress_handler = None
        self._solve_details = None
        self.abs = np.vectorize(self._abs_scalar, otypes=[object])

        if not verbose:
            self._scip.hideOutput()

        timelimit_s = timelimit.to_value(u.s)
        if timelimit_s < 1e75:
            self._scip.setParam("limits/time", timelimit_s)
            # SCIP finds no feasible solution at all on these models with its
            # default settings, so it is pushed towards feasibility whenever
            # the search is time limited.
            self._scip.setEmphasis(SCIP_PARAMEMPHASIS.FEASIBILITY)
            self._scip.setHeuristics(SCIP_PARAMSETTING.AGGRESSIVE)

        if np.isfinite(memory):
            self._scip.setParam("limits/memory", memory.to_value(u.MiB))

        if lowercutoff is not None:
            # Give up once SCIP can prove that nothing better than the cutoff
            # exists.
            self._scip.setObjlimit(lowercutoff)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._scip.freeProb()

    def _new_aux_var(self, lb=None, ub=None):
        return self._scip.addVar(lb=lb, ub=ub)

    def _create_var_list(self, tp, size, lb, ub):
        vtype = {
            "binary": "B",
            "continuous": "C",
            "integer": "I",
            "semicontinuous": "C",
            "semiinteger": "I",
        }[tp]
        semi = tp.startswith("semi")
        vars = []
        for i in range(size):
            lb_i = lb[i] if isinstance(lb, np.ndarray) else lb
            ub_i = ub[i] if isinstance(ub, np.ndarray) else ub
            # SCIP reads a lower bound of None as negative infinity, while the
            # other backends default a decision variable to zero.
            if lb_i is None:
                lb_i = 0.0
            if semi:
                # SCIP has no semi-continuous variable type, so the "zero or
                # within bounds" behaviour is built from an indicator binary.
                var = self._scip.addVar(vtype=vtype, lb=0.0, ub=ub_i)
                switch = self._scip.addVar(vtype="B")
                if ub_i is not None:
                    self._scip.addCons(var <= ub_i * switch)
                if lb_i:
                    self._scip.addCons(var >= lb_i * switch)
            elif vtype == "B":
                var = self._scip.addVar(vtype="B")
            else:
                var = self._scip.addVar(vtype=vtype, lb=lb_i, ub=ub_i)
            vars.append(var)
        return [SCIPVarProxy(v, self) for v in vars]

    def binary_var(self, name=None):
        return SCIPVarProxy(self._scip.addVar(vtype="B"), self)

    def continuous_var(self, name=None, lb=None, ub=None):
        return SCIPVarProxy(
            self._scip.addVar(lb=0.0 if lb is None else lb, ub=ub), self
        )

    def integer_var(self, name=None, lb=None, ub=None):
        return SCIPVarProxy(
            self._scip.addVar(vtype="I", lb=0.0 if lb is None else lb, ub=ub), self
        )

    def _abs_scalar(self, expr):
        """Return ``|expr|`` split into its positive and negative parts.

        The two parts are held in an SOS1 set, so at most one is nonzero and
        their sum is the absolute value. This needs no big-M bound.
        """
        expr = _unwrap(expr)
        if isinstance(expr, (int, float)):
            return abs(expr)
        # The expression is split into its positive and negative parts, at most
        # one of which may be nonzero, so their sum is the absolute value. An
        # indicator formulation of the same disjunction searches far worse.
        lo, hi = self._expr_bounds(expr)
        pos = self._scip.addVar(lb=0.0, ub=max(hi, 0.0) if hi < _INFINITY else None)
        neg = self._scip.addVar(lb=0.0, ub=max(-lo, 0.0) if lo > -_INFINITY else None)
        self._scip.addCons(expr == pos - neg)
        self._scip.addConsSOS1([pos, neg])
        return SCIPExprProxy(pos + neg, self)

    def _expr_bounds(self, expr):
        """Range a linear expression can reach, given its variables' bounds."""
        try:
            return expr.getLbOriginal(), expr.getUbOriginal()
        except AttributeError:
            pass
        terms = getattr(expr, "terms", None)
        if terms is None:
            return -_INFINITY, _INFINITY
        lo = hi = 0.0
        for term, coeff in terms.items():
            if not term:
                lo += coeff
                hi += coeff
                continue
            if len(term) != 1:
                return -_INFINITY, _INFINITY
            var_lb, var_ub = term[0].getLbOriginal(), term[0].getUbOriginal()
            if var_lb <= -_INFINITY or var_ub >= _INFINITY:
                return -_INFINITY, _INFINITY
            lo += min(coeff * var_lb, coeff * var_ub)
            hi += max(coeff * var_lb, coeff * var_ub)
        return lo, hi

    def add_constraint_(self, ct, name=None):
        """Add a single constraint to the model."""
        ct = _unwrap(ct)
        if isinstance(ct, SCIPEqualityProxy):
            self._scip.addCons(ct._lhs == ct._rhs)
        else:
            self._scip.addCons(ct)

    def add_constraints_(self, cts, names=None):
        """Add any number of constraints to the model."""
        for ct in _flatten_constraints(cts):
            self.add_constraint_(ct)

    def add_indicators(self, binary_vars, cts, true_values=1, names=None):
        """Add any number of indicator constraints to the model."""
        binary_vars = np.asarray(binary_vars).ravel()
        cts = np.asarray(cts).ravel()
        true_values = np.broadcast_to(true_values, len(binary_vars)).ravel()
        for bvar, ct, tv in zip(binary_vars, cts, true_values):
            self._add_indicator(_unwrap(bvar), bool(tv), ct)

    def _add_indicator(self, binvar, binval, constraint):
        """Add ``binvar == binval -> constraint``.

        SCIP's indicator constraints take a ``<=`` inequality, so a ``>=``
        constraint is negated and an equality is split into two indicators.
        """
        for cons in self._as_upper_bounds(constraint):
            self._scip.addConsIndicator(cons, binvar, activeone=binval)

    @staticmethod
    def _as_upper_bounds(constraint):
        """Rewrite a constraint as one or two ``expr <= rhs`` inequalities."""
        cons = _unwrap(constraint)
        if isinstance(cons, SCIPEqualityProxy):
            lhs, rhs = cons._lhs, cons._rhs
            return [lhs - rhs <= 0, rhs - lhs <= 0]
        lhs = cons.expr
        if cons._lhs is not None and cons._rhs is not None:
            return [lhs <= cons._rhs, -lhs <= -cons._lhs]
        if cons._rhs is not None:
            return [lhs <= cons._rhs]
        return [-lhs <= -cons._lhs]

    def add_indicator_constraints(self, indcts):
        """Add indicator constraints from ``(var == val) >> constr``."""
        for ic in np.asarray(indcts).ravel():
            if not isinstance(ic, SCIPIndicatorProxy):
                raise TypeError(f"Expected SCIPIndicatorProxy, got {type(ic)}")
            self._add_indicator(ic._binvar, bool(ic._binval), ic._constr)

    def add_indicator_constraints_(self, indcts):
        """Same as add_indicator_constraints (batch version)."""
        self.add_indicator_constraints(indcts)

    def add_sos1(self, vars):
        """Add an SOS1 constraint: at most one variable may be nonzero."""
        self._scip.addConsSOS1([_unwrap(v) for v in vars])

    def add_user_cut_constraint(self, ct):
        """Add a user cut (SCIP takes it as an ordinary constraint)."""
        self.add_constraint_(ct)

    def maximize(self, expr):
        self._scip.setObjective(_unwrap(expr), "maximize")

    def minimize(self, expr):
        self._scip.setObjective(_unwrap(expr), "minimize")

    def _extremum(self, args, upper):
        """Return a variable pinned to the max (or min) of ``args``.

        One indicator per argument, with exactly one active, forces the
        auxiliary variable onto whichever argument is extremal.
        """
        exprs = [_unwrap(a) for a in args]
        # Without an explicit range the auxiliary variable is free in the LP
        # relaxation, where the indicators below are inactive, and the whole
        # constraint becomes vacuous.
        bounds = [self._var_bounds(expr) for expr in exprs]
        lo = min(b[0] for b in bounds)
        hi = max(b[1] for b in bounds)
        aux = self._new_aux_var(lb=lo, ub=hi)
        switches = []
        for expr, (expr_lo, expr_hi) in zip(exprs, bounds):
            self._scip.addCons(aux >= expr if upper else aux <= expr)
            switch = self._scip.addVar(vtype="B")
            # A big-M disjunction relaxes better than an indicator, which
            # constrains nothing until its binary is fixed. M is the largest
            # gap the operand ranges allow, so it stays tight.
            big_m = (hi - expr_lo) if upper else (expr_hi - lo)
            if big_m < _INFINITY:
                self._scip.addCons(
                    aux - expr <= big_m * (1 - switch)
                    if upper
                    else expr - aux <= big_m * (1 - switch)
                )
            else:
                self._scip.addConsIndicator(
                    aux - expr <= 0 if upper else expr - aux <= 0, switch
                )
            switches.append(switch)
        self._scip.addCons(scip.quicksum(switches) == 1)
        return np.asarray(SCIPVarProxy(aux, self)).view(VariableArray)

    @staticmethod
    def _var_bounds(expr):
        """Bounds of a variable, or an infinite range for a general expression."""
        try:
            return expr.getLbOriginal(), expr.getUbOriginal()
        except AttributeError:
            return -_INFINITY, _INFINITY

    def min(self, *args):
        """Return the minimum of the given variables/expressions."""
        return self._extremum(args, upper=False)

    def max(self, *args):
        """Return the maximum of the given variables/expressions."""
        return self._extremum(args, upper=True)

    def sum(self, args):
        return SCIPExprProxy(scip.quicksum(_unwrap(a) for a in args), self)

    def sum_vars_all_different(self, vars):
        return SCIPExprProxy(
            scip.quicksum(_unwrap(v) for v in np.asarray(vars).ravel()), self
        )

    def scal_prod_vars_all_different(self, vars, coeffs):
        return SCIPExprProxy(
            scip.quicksum(
                float(c) * _unwrap(v)
                for v, c in zip(np.asarray(vars).ravel(), np.asarray(coeffs).ravel())
            ),
            self,
        )

    def piecewise(self, preslope, breakpoints, postslope):
        """Return a callable creating a piecewise linear constraint.

        The curve is a convex combination of its breakpoints, with the weights
        held in an SOS2 set so that only two adjacent ones may be nonzero.
        """

        def apply(var):
            var = _unwrap(var)
            xpts = [float(bp[0]) for bp in breakpoints]
            ypts = [float(bp[1]) for bp in breakpoints]

            # A convex combination is confined to the breakpoints, so the
            # slopes are extended all the way to the bounds of the variable
            # rather than by a nominal step.
            lb, ub = var.getLbOriginal(), var.getUbOriginal()
            if xpts and preslope != 0 and lb > -_INFINITY and lb < xpts[0]:
                ypts.insert(0, ypts[0] - preslope * (xpts[0] - lb))
                xpts.insert(0, lb)
            if xpts and postslope != 0 and ub < _INFINITY and ub > xpts[-1]:
                ypts.append(ypts[-1] + postslope * (ub - xpts[-1]))
                xpts.append(ub)

            weights = [self._scip.addVar(lb=0.0, ub=1.0) for _ in xpts]
            self._scip.addCons(scip.quicksum(weights) == 1)
            self._scip.addConsSOS2(weights)
            self._scip.addCons(
                var == scip.quicksum(w * x for w, x in zip(weights, xpts))
            )
            aux = self._new_aux_var()
            self._scip.addCons(
                aux == scip.quicksum(w * y for w, y in zip(weights, ypts))
            )
            return SCIPVarProxy(aux, self)

        return apply

    def add_progress_listener(self, recorder):
        """Record solver progress whenever the incumbent improves."""
        self._progress_handler = _ProgressEventHandler(recorder)
        self._scip.includeEventhdlr(
            self._progress_handler, "m4optProgress", "records MILP progress"
        )

    def solve(self, **kwargs):
        with status("solving MILP model"):
            self._scip.optimize()

        status_str = self._scip.getStatus()
        self._solve_details = SolveDetails(
            status=status_str, time=self._scip.getSolvingTime()
        )
        if self._scip.getNSols() == 0:
            return None
        return SCIPSolveSolution(self._scip, self._solve_details)

    @property
    def number_of_variables(self):
        """Number of decision variables in the model."""
        return self._scip.getNVars()

    @property
    def number_of_constraints(self):
        """Number of constraints in the model."""
        return self._scip.getNConss()

    @property
    def objective_expr(self):
        """The objective, wrapped to report its number of terms."""
        return SCIPObjectiveProxy(self._scip.getObjective())

    @property
    def best_bound(self):
        """Get the best bound from the last solve."""
        return self._scip.getDualbound()

    @property
    def solve_details(self):
        """Get solve details from the last solve."""
        if self._solve_details is None:
            return SolveDetails(status="not solved", time=0.0)
        return self._solve_details

    @property
    def objective_value(self):
        """Get the objective value from the last solve."""
        return self._scip.getObjVal()

    def to_stream(self, out_file: BufferedWriter):
        """Write the model to a stream, inferring the format from its name."""
        out_path = Path(getattr(out_file, "name", "model.lp"))
        suffixes = [suffix.lower() for suffix in out_path.suffixes]
        should_gzip = bool(suffixes) and suffixes[-1] == ".gz"
        format_suffix = (
            suffixes[-2]
            if should_gzip and len(suffixes) > 1
            else (suffixes[-1] if suffixes else "")
        )
        valid = (".lp", ".mps", ".cip")
        if format_suffix not in valid:
            valid_extensions = [*valid, *(f"{ext}.gz" for ext in valid)]
            raise ValueError(
                f'Invalid model filename "{out_path}". The extension must be one '
                f"of the following: {' '.join(valid_extensions)}"
            )
        if should_gzip:
            with NamedTemporaryFile(suffix=format_suffix) as temp_file:
                self._scip.writeProblem(temp_file.name, verbose=False)
                with (
                    GzipFile(out_path.stem, "wb", fileobj=out_file) as zip_file,
                    open(temp_file.name, "rb") as written,
                ):
                    copyfileobj(written, zip_file)
        else:
            self._scip.writeProblem(str(out_path), verbose=False)


class SCIPObjectiveProxy:
    """Wrap a SCIP objective to expose docplex's term count."""

    def __init__(self, expr):
        self._expr = expr

    def number_of_terms(self):
        return len(self._expr.terms)


class SCIPSolveSolution:
    """The solution of a solved SCIP model."""

    def __init__(self, scip_model, solve_details):
        self._scip = scip_model
        self._solve_details = solve_details

    def get_values(self, var_seq):
        """Get solution values for multidimensional arrays of variables."""
        var_seq = np.asarray(var_seq)
        values = np.array([self._scip.getVal(_unwrap(v)) for v in var_seq.ravel()])
        return values.reshape(var_seq.shape)

    def get_objective_value(self):
        """Get the objective value of this solution."""
        return self._scip.getObjVal()


for _tp in _VARIABLE_TYPES:
    add_var_array_method(SCIPModel, _tp)
del _tp
