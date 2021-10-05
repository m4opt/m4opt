from gurobipy import GurobiError, GRB
import pytest


def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi problem size."""
    try:
        item.runtest()
    except GurobiError as e:
        if e.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
            pytest.skip('requires full version of Gurobi')
        raise
