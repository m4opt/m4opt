import pytest
from docplex.mp.utils import DOcplexLimitsExceeded


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi or CPLEX problem size."""
    outcome = yield
    if outcome.excinfo is not None and issubclass(
        outcome.excinfo[0], DOcplexLimitsExceeded
    ):
        pytest.skip("requires full version of CPLEX")
