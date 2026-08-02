import pytest
from docplex.mp.utils import DOcplexLimitsExceeded


def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi or CPLEX problem size."""
    try:
        item.runtest()
    except DOcplexLimitsExceeded:
        pytest.skip("requires full version of CPLEX")
