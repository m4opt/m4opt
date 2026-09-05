"""Test production problem size limits for MILP solvers."""

from tempfile import gettempdir

import numpy as np
import pytest
from astropy import units as u

from .._milp import ProgressDataRecorder, _get_backend
from ..milp import _VARIABLE_TYPES, Model, VariableArray

problem_size_limits = pytest.mark.parametrize(
    "num_vars", [pytest.param(1000, id="small"), pytest.param(10000, id="big")]
)

backend = _get_backend()
cplex_only = pytest.mark.skipif(backend != "cplex", reason="CPLEX-only test")


@pytest.fixture
def m():
    with Model() as m:
        yield m


@pytest.fixture(params=_VARIABLE_TYPES)
def add_vars(m, request):
    return getattr(m, f"{request.param}_vars")


@problem_size_limits
def test_problem_size(m, num_vars):
    """Test that the solver works with small and big problems."""
    m.binary_vars(num_vars)
    m.solve()


@cplex_only
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
    assert m.best_bound == m.objective_value == 1


def test_add_var_array(m, add_vars):
    """Test convenience functions for adding arrays of decision variables."""
    result = add_vars()
    assert not isinstance(result, np.ndarray)

    result = add_vars((6, 4))
    assert result.shape == (6, 4)

    result = add_vars((6, 4), lb=np.full((6, 4), 0.5), ub=np.full((6, 4), 1))
    assert result.shape == (6, 4)

    if backend == "scip":
        # SCIP has no semi-continuous variable type, so m4opt emulates it with
        # an extra binary per variable.
        assert m.number_of_variables >= 49
    else:
        assert m.number_of_variables == 49


@pytest.mark.xfail(reason="Does not yet pass")
def test_sum(m, add_vars):
    """Test Numpy sum method on decision variables."""
    x = add_vars(3)
    assert x.sum().equals(m.sum(x))
    assert np.sum(x).equals(m.sum(x))


@pytest.mark.parametrize("rhs_shape", ((), 2, (3, 2)))
@pytest.mark.parametrize(
    "expr",
    (
        "x >= y",
        "x <= y",
        "x == y",
        "x + 5 <= y",
        "y + 5 <= 0",
        "y + 5 <= x[0][0]",
        "x + y <= 0",
        "x - y <= 0",
        "m.min(*x.ravel()) <= 0",
        "m.max(*x.ravel()) <= 0",
    ),
)
def test_operators(m, add_vars, rhs_shape, expr):
    """Test adding constraints by broadcasting variables."""
    constraint = eval(
        expr, None, {"m": m, "x": add_vars((3, 2)), "y": add_vars(rhs_shape)}
    )
    m.add_constraints_(constraint)


@pytest.mark.parametrize("rhs_shape", ((), 2, (3, 2)))
def test_broadcast_indicator(m, add_vars, rhs_shape):
    """Test adding indicator constraints by broadcasting variables."""
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
        ".lp.gz",
        ".mps.gz",
        # SAV is a CPLEX-native format that Gurobi cannot write.
        pytest.param(".sav", marks=cplex_only),
        pytest.param(".sav.gz", marks=cplex_only),
    ],
)
def test_to_stream(m, suffix, tmp_path):
    x = m.binary_var()
    m.maximize(x)
    with (tmp_path / "model").with_suffix(suffix).open("wb") as f:
        m.to_stream(f)


def test_arithmetic(m):
    """Variables support the arithmetic that constraint expressions are built from."""
    x = m.continuous_vars(3, lb=0, ub=1)
    m.add_constraints_(2 * x <= 2)
    m.add_constraints_(x * 2 <= 2)
    m.add_constraints_(-x <= 0)
    m.add_constraints_(x - 5 <= 0)
    m.add_constraints_(x + 5 >= 0)
    m.maximize(m.sum(x.ravel()))
    assert m.solve().get_objective_value() == pytest.approx(3)


def test_minimize(m):
    x = m.continuous_vars(3, lb=1, ub=5)
    m.minimize(m.sum(x.ravel()))
    assert m.solve().get_objective_value() == pytest.approx(3)


def test_abs(m):
    """Absolute value bounds a variable from both sides."""
    x = m.continuous_vars(2, lb=-10, ub=10)
    m.add_constraints_(m.abs(x) <= 3)
    m.maximize(m.sum(x.ravel()))
    assert m.solve().get_objective_value() == pytest.approx(6)


def test_scal_prod(m):
    x = m.binary_vars(4)
    m.add_constraint_(m.sum_vars_all_different(x.ravel()) <= 2)
    m.maximize(m.scal_prod_vars_all_different(x.ravel(), [1.0, 2.0, 3.0, 4.0]))
    assert m.solve().get_objective_value() == pytest.approx(7)


def test_sos1(m):
    """An SOS1 set allows at most one nonzero member."""
    x = m.continuous_vars(3, lb=0, ub=1)
    m.add_sos1(list(x.ravel()))
    m.maximize(m.sum(x.ravel()))
    assert m.solve().get_objective_value() == pytest.approx(1)


def test_piecewise(m):
    """A piecewise linear function saturates beyond its last breakpoint."""
    x = m.continuous_vars(lb=0, ub=10)
    y = m.continuous_vars(lb=0, ub=100)
    m.add_constraint_(y <= m.piecewise(0, [(0, 0), (5, 5), (10, 5)], 0)(x))
    m.maximize(y)
    assert m.solve().get_objective_value() == pytest.approx(5)


def test_solve_details(m):
    x = m.binary_var()
    m.maximize(x)
    m.solve()
    assert m.solve_details.status
    assert m.solve_details.time >= 0


def test_progress_listener(m):
    """A progress listener receives the solver's search statistics."""
    recorder = ProgressDataRecorder()
    m.add_progress_listener(recorder)
    n = 60
    x = m.binary_vars(n)
    weights = [1 + (i * 7) % 23 for i in range(n)]
    m.add_constraint_(m.scal_prod_vars_all_different(x.ravel(), weights) <= 137)
    m.maximize(m.scal_prod_vars_all_different(x.ravel(), [w + 0.5 for w in weights]))
    m.solve()
    assert isinstance(recorder.recorded, list)


def test_piecewise_slopes(m):
    """Slopes extend the function beyond its first and last breakpoints."""
    x = m.continuous_vars(lb=-5, ub=20)
    y = m.continuous_vars(lb=-100, ub=100)
    m.add_constraint_(x >= 15)
    m.add_constraint_(y <= m.piecewise(2, [(0, 0), (10, 10)], 3)(x))
    m.maximize(y)
    # Beyond the last breakpoint the value grows with the postslope.
    assert m.solve().get_objective_value() > 10


def test_constraint_truth_value_is_ambiguous(m):
    """A constraint must be added to the model, not evaluated as a bool."""
    x = m.binary_vars(2)
    with pytest.raises(TypeError):
        bool(x[0] == x[1])


def test_set_backend_rejects_unknown_name():
    from .._milp import set_backend

    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("nonesuch")


def test_solution_values_match_shape(m):
    """Solution values come back shaped like the variable array."""
    x = m.continuous_vars((3, 2), lb=1, ub=1)
    m.maximize(m.sum(x.ravel()))
    solution = m.solve()
    assert solution.get_values(x).shape == (3, 2)
    assert solution.get_objective_value() == pytest.approx(6)


def test_variable_bounds(m):
    """Variables report the bounds they were created with."""
    x = m.continuous_vars(3, lb=2, ub=7)
    assert [v.lb for v in x] == [2, 2, 2]
    assert [v.ub for v in x] == [7, 7, 7]


# Semi-continuous and semi-integer variables are excluded: their lower bound is
# a threshold rather than a bound, and the backends differ on its default.
@pytest.mark.parametrize("vartype", ["continuous", "integer"])
def test_default_variable_bounds(m, vartype):
    """A variable created without bounds is non-negative, not free."""
    x = getattr(m, f"{vartype}_vars")(3)
    assert all(v.lb == 0 for v in x)
