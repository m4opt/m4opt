import pytest


def pytest_runtest_call(item):
    """Skip tests that exceed the Gurobi or CPLEX problem size."""
    try:
        item.runtest()
    except Exception as e:
        # DOcplex community edition limit
        if type(e).__name__ == "DOcplexLimitsExceeded":
            pytest.skip("requires full version of CPLEX")
        # Gurobi community/restricted license limit or license error
        if type(e).__name__ == "GurobiError" and (
            "size-limited" in str(e).lower() or "license" in str(e).lower()
        ):
            pytest.skip("requires full version of Gurobi")
        raise
