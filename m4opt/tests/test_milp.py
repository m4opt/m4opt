"""Test production problem size limits for MILP solvers."""

import docplex.mp.model
import gurobipy
import numpy as np
import pytest
from astropy import units as u

from ..milp import Model

problem_size_limits = pytest.mark.parametrize(
    "num_vars", [pytest.param(1000, id="small"), pytest.param(10000, id="big")]
)


@problem_size_limits
def test_cplex(num_vars):
    """Test that CPLEX solver works with small and big problems."""
    m = docplex.mp.model.Model()
    m.binary_var_list(num_vars)
    m.solve()


@problem_size_limits
def test_gurobi(num_vars):
    """Test that Gurobi solver works with small and big problems."""
    m = gurobipy.Model()
    m.addMVar(num_vars)
    m.optimize()


def test_cplex_parameters():
    """Test configuration of CPLEX solver parameters."""
    m = Model()
    assert m.context.cplex_parameters.emphasis.mip.value == 0
    assert m.context.cplex_parameters.mip.pool.capacity.value == 0
    assert m.context.cplex_parameters.parallel.value == -1
    assert m.context.cplex_parameters.threads.value == 0
    assert m.context.cplex_parameters.timelimit.value == 1e75
    assert m.context.solver.log_output

    m = Model(timelimit=1 * u.minute, jobs=3)
    assert m.context.cplex_parameters.emphasis.mip.value == 1
    assert m.context.cplex_parameters.mip.pool.capacity.value == 0
    assert m.context.cplex_parameters.parallel.value == -1
    assert m.context.cplex_parameters.threads.value == 3
    assert m.context.cplex_parameters.timelimit.value == 60
    assert m.context.solver.log_output


@pytest.mark.parametrize(
    "tp", ["binary", "continuous", "integer", "semicontinuous", "semiinteger"]
)
def test_cplex_add_var_array(tp):
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

    assert getattr(m, f"number_of_{tp}_variables") == 49
