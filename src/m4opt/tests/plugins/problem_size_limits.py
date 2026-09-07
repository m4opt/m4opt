import pytest
from docplex.mp.utils import DOcplexLimitsExceeded


def _skip_if_problem_too_large(outcome):
    if outcome.excinfo is not None and issubclass(
        outcome.excinfo[0], DOcplexLimitsExceeded
    ):
        pytest.skip("requires full version of CPLEX")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    """Skip tests whose fixtures exceed the CPLEX problem size."""
    outcome = yield
    _skip_if_problem_too_large(outcome)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Skip tests that exceed the CPLEX problem size."""
    outcome = yield
    _skip_if_problem_too_large(outcome)
