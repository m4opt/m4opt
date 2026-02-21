"""Test production problem size limits for MILP solvers."""

from tempfile import gettempdir

import numpy as np
import pytest
from astropy import units as u

from ..milp import Model, VariableArray
from .._milp import _get_backend

problem_size_limits = pytest.mark.parametrize(
    "num_vars", [pytest.param(1000, id="small"), pytest.param(10000, id="big")]
)

_backend = _get_backend()
_is_cplex = _backend == "cplex"
_is_gurobi = _backend == "gurobi"

cplex_only = pytest.mark.skipif(not _is_cplex, reason="CPLEX-only test")
gurobi_only = pytest.mark.skipif(not _is_gurobi, reason="Gurobi-only test")


@cplex_only
@problem_size_limits
def test_cplex(num_vars):
    """Test that CPLEX solver works with small and big problems."""
    import docplex.mp.model

    m = docplex.mp.model.Model()
    m.binary_var_list(num_vars)
    m.solve()


@gurobi_only
@problem_size_limits
def test_gurobi(num_vars):
    """Test that Gurobi solver works with small and big problems."""
    import gurobipy as gp

    m = gp.Model()
    m.addVars(num_vars, vtype=gp.GRB.BINARY)
    m.optimize()


@cplex_only
def test_cplex_parameters():
    """Test configuration of CPLEX solver parameters."""
    m = Model()
    assert m.context.cplex_parameters.mip.pool.capacity.value == 0
    assert m.context.cplex_parameters.mip.strategy.file.value == 1
    assert m.context.cplex_parameters.parallel.value == -1
    assert m.context.cplex_parameters.threads.value == 0
    assert m.context.cplex_parameters.timelimit.value == 1e75
    assert m.context.cplex_parameters.workmem.value == 2048
    assert m.context.cplex_parameters.workdir.value == gettempdir()
    assert m.context.solver.log_output

    m = Model(timelimit=1 * u.minute, jobs=3)
    assert m.context.cplex_parameters.mip.pool.capacity.value == 0
    assert m.context.cplex_parameters.mip.strategy.file.value == 1
    assert m.context.cplex_parameters.parallel.value == -1
    assert m.context.cplex_parameters.threads.value == 3
    assert m.context.cplex_parameters.timelimit.value == 60
    assert m.context.cplex_parameters.workmem.value == 2048
    assert m.context.cplex_parameters.workdir.value == gettempdir()
    assert m.context.solver.log_output

    m = Model(timelimit=1 * u.minute, jobs=3, memory=5 * u.GiB)
    assert m.context.cplex_parameters.mip.pool.capacity.value == 0
    assert m.context.cplex_parameters.mip.strategy.file.value == 3
    assert m.context.cplex_parameters.parallel.value == -1
    assert m.context.cplex_parameters.threads.value == 3
    assert m.context.cplex_parameters.timelimit.value == 60
    assert m.context.cplex_parameters.workmem.value == 5120
    assert m.context.cplex_parameters.workdir.value == gettempdir()
    assert m.context.solver.log_output


@gurobi_only
def test_gurobi_parameters():
    """Test configuration of Gurobi solver parameters."""
    m = Model()
    assert m._grb.Params.Threads == 0
    assert m._grb.Params.LogToConsole == 1

    m = Model(timelimit=1 * u.minute, jobs=3)
    assert m._grb.Params.Threads == 3
    assert m._grb.Params.TimeLimit == 60
    assert m._grb.Params.MIPFocus == 1

    m = Model(timelimit=1 * u.minute, jobs=3, memory=5 * u.GiB)
    assert m._grb.Params.Threads == 3
    assert m._grb.Params.TimeLimit == 60
    assert m._grb.Params.NodefileStart == pytest.approx(5.0)


def test_best_bound():
    m = Model()
    x = m.binary_vars()
    m.maximize(x)
    solution = m.solve()
    assert solution is not None
    assert m.best_bound == pytest.approx(1)
    assert m.objective_value == pytest.approx(1)


@pytest.mark.parametrize(
    "tp", ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]
)
def test_add_var_array(tp):
    """Test convenience functions for adding arrays of decision variables."""
    m = Model()

    result = getattr(m, f"{tp}_vars")()
    assert not isinstance(result, np.ndarray)

    result = getattr(m, f"{tp}_vars")((6, 4))
    assert result.shape == (6, 4)

    result = getattr(m, f"{tp}_vars")(
        (6, 4), lb=np.full((6, 4), 0.5), ub=np.full((6, 4), 1)
    )
    assert result.shape == (6, 4)


@cplex_only
@pytest.mark.parametrize(
    "tp", ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]
)
def test_cplex_add_var_array_count(tp):
    """Test that CPLEX tracks variable counts correctly."""
    m = Model()
    getattr(m, f"{tp}_vars")()
    getattr(m, f"{tp}_vars")((6, 4))
    getattr(m, f"{tp}_vars")((6, 4), lb=np.full((6, 4), 0.5), ub=np.full((6, 4), 1))
    assert getattr(m, f"number_of_{tp}_variables") == 49


@pytest.mark.parametrize(
    "tp", ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]
)
@pytest.mark.parametrize(
    "rhs_shape",
    ((), 2, (3, 2)),
)
@pytest.mark.parametrize(
    "expr",
    (
        "x >= y",
        "x <= y",
        "x == y",
        "x + 5 <= y",
        "x + y <= 0",
        "x - y <= 0",
        "m.min(*x.ravel()) <= 0",
        "m.max(*x.ravel()) <= 0",
    ),
)
def test_operators(tp, rhs_shape, expr):
    """Test adding constraints by broadcasting variables."""
    m = Model()
    add_vars = getattr(m, f"{tp}_vars")
    constraint = eval(expr, None, dict(m=m, x=add_vars((3, 2)), y=add_vars(rhs_shape)))
    assert isinstance(constraint, VariableArray)
    m.add_constraints_(constraint)


@pytest.mark.parametrize(
    "tp", ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]
)
@pytest.mark.parametrize(
    "rhs_shape",
    ((), 2, (3, 2)),
)
def test_broadcast_indicator(tp, rhs_shape):
    """Test adding indicator constraints by broadcasting variables."""
    m = Model()
    add_vars = getattr(m, f"{tp}_vars")
    x = m.binary_vars((3, 2))
    y = add_vars(rhs_shape)
    constraint = (x == 1) >> (y >= 0)
    assert isinstance(constraint, VariableArray)
    m.add_indicator_constraints(constraint)
    m.add_indicator_constraints_(constraint)


@pytest.mark.parametrize(
    "suffix",
    [
        ".lp",
        ".mps",
        pytest.param(".sav", marks=cplex_only),
        ".lp.gz",
        ".mps.gz",
        pytest.param(".sav.gz", marks=cplex_only),
    ],
)
def test_to_stream(suffix, tmp_path):
    m = Model()
    x = m.binary_vars()
    m.maximize(x)
    with (tmp_path / "model").with_suffix(suffix).open("wb") as f:
        m.to_stream(f)
