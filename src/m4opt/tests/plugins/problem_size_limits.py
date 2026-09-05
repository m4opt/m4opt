import pytest

try:
    from docplex.mp.utils import DOcplexLimitsExceeded
except ImportError:
    DOcplexLimitsExceeded = ()

try:
    from gurobipy import GurobiError
except ImportError:
    GurobiError = ()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi or CPLEX problem size."""
    outcome = yield
    if outcome.excinfo is None:
        return
    exception_type, exception = outcome.excinfo[:2]
    if DOcplexLimitsExceeded and issubclass(exception_type, DOcplexLimitsExceeded):
        pytest.skip("requires full version of CPLEX")
    # Gurobi raises the same error type for every failure, so the message is
    # the only thing that distinguishes a size limit from a real problem.
    if (
        GurobiError
        and issubclass(exception_type, GurobiError)
        and ("size-limited" in str(exception) or "license" in str(exception).lower())
    ):
        pytest.skip("requires full version of Gurobi")
