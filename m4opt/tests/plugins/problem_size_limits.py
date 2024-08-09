import pytest
from docplex.mp.utils import DOcplexLimitsExceeded
from gurobipy import GRB, GurobiError


def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi or CPLEX problem size."""
    try:
        item.runtest()
    except GurobiError as e:
        if e.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
            pytest.skip("requires full version of Gurobi")
        raise
    except DOcplexLimitsExceeded:
        pytest.skip("requires full version of CPLEX")
