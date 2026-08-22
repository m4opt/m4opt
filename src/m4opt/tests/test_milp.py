"""Test production problem size limits for MILP solvers."""

from tempfile import gettempdir

import numpy as np
import pytest
from astropy import units as u

from ..milp import _VARIABLE_TYPES, Model, VariableArray

problem_size_limits = pytest.mark.parametrize(
    "num_vars", [pytest.param(1000, id="small"), pytest.param(10000, id="big")]
)


@pytest.fixture
def m():
    with Model() as m:
        yield m


@pytest.fixture(params=_VARIABLE_TYPES)
def add_vars(m, request):
    return getattr(m, f"{request.param}_vars")


@problem_size_limits
def test_cplex(m, num_vars):
    """Test that CPLEX solver works with small and big problems."""
    m.binary_var_list(num_vars)
    m.solve()


def test_cplex_parameters():
    """Test configuration of CPLEX solver parameters."""
    with Model() as m:
        assert m.context.cplex_parameters.mip.pool.capacity.value == 0
        assert m.context.cplex_parameters.mip.strategy.file.value == 1
        assert m.context.cplex_parameters.parallel.value == -1
        assert m.context.cplex_parameters.threads.value == 0
        assert m.context.cplex_parameters.timelimit.value == 1e75
        assert m.context.cplex_parameters.workmem.value == 2048
        assert m.context.cplex_parameters.workdir.value == gettempdir()
        assert m.context.solver.log_output

    with Model(timelimit=1 * u.minute, jobs=3) as m:
        assert m.context.cplex_parameters.mip.pool.capacity.value == 0
        assert m.context.cplex_parameters.mip.strategy.file.value == 1
        assert m.context.cplex_parameters.parallel.value == -1
        assert m.context.cplex_parameters.threads.value == 3
        assert m.context.cplex_parameters.timelimit.value == 60
        assert m.context.cplex_parameters.workmem.value == 2048
        assert m.context.cplex_parameters.workdir.value == gettempdir()
        assert m.context.solver.log_output

    with Model(timelimit=1 * u.minute, jobs=3, memory=5 * u.GiB) as m:
        assert m.context.cplex_parameters.mip.pool.capacity.value == 0
        assert m.context.cplex_parameters.mip.strategy.file.value == 3
        assert m.context.cplex_parameters.parallel.value == -1
        assert m.context.cplex_parameters.threads.value == 3
        assert m.context.cplex_parameters.timelimit.value == 60
        assert m.context.cplex_parameters.workmem.value == 5120
        assert m.context.cplex_parameters.workdir.value == gettempdir()
        assert m.context.solver.log_output


def test_best_bound(m):
    x = m.binary_var()
    m.maximize(x)
    m.solve()
    assert m.best_bound == m.solve_details.best_bound == m.objective_value == 1


def test_cplex_add_var_array(m, add_vars):
    """Test convenience functions for adding arrays of decision variables."""
    result = add_vars()
    assert not isinstance(result, np.ndarray)

    result = add_vars((6, 4))
    assert result.shape == (6, 4)

    result = add_vars((6, 4), lb=np.full((6, 4), 0.5), ub=np.full((6, 4), 1))
    assert result.shape == (6, 4)

    assert m.number_of_variables == 49


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
def test_cplex_operators(m, add_vars, rhs_shape, expr):
    """Test adding constraints by broadcasting variables."""
    constraint = eval(
        expr, None, {"m": m, "x": add_vars((3, 2)), "y": add_vars(rhs_shape)}
    )
    assert isinstance(constraint, VariableArray)
    m.add_constraints_(constraint)


@pytest.mark.parametrize(
    "rhs_shape",
    ((), 2, (3, 2)),
)
def test_cplex_broadcast_indicator(m, add_vars, rhs_shape):
    """Test adding indicator constraints by broadcasting variables."""
    x = m.binary_vars((3, 2))
    y = add_vars(rhs_shape)
    constraint = (x == 1) >> (y >= 0)
    assert isinstance(constraint, VariableArray)
    m.add_indicator_constraints(constraint)
    m.add_indicator_constraints_(constraint)


@pytest.mark.parametrize(
    "suffix", [".lp", ".mps", ".sav", ".lp.gz", ".mps.gz", ".sav.gz"]
)
def test_to_stream(m, suffix, tmp_path):
    x = m.binary_var()
    m.maximize(x)
    with (tmp_path / "model").with_suffix(suffix).open("wb") as f:
        m.to_stream(f)
