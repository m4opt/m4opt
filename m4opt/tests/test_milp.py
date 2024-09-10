"""Test production problem size limits for MILP solvers."""

import docplex.mp.model
import gurobipy
import numpy as np
import pytest

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

    assert getattr(m, f"number_of_{tp}_variables") == 25