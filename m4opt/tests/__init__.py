from gurobipy import GurobiError, GRB, Model
import pytest


def gurobi_problem_size_is_limited():
    """Determine if the Gurobi problem size is limited by the license.

    Notes
    -----
    According to the `Gurobi EULA <https://www.gurobi.com/eula>`_, the free
    evaluation license has a problem size limit of 2000 variables. This
    function tests if a full license is present by attempting to solve a
    trivial problem with just one too many variables for the free license.

    """
    m = Model()
    m.addMVar(2001)

    try:
        m.optimize()
    except GurobiError as e:
        if e.errno == GRB.Error.SIZE_LIMIT_EXCEEDED:
            return True
        raise

    return False


skip_if_gurobi_problem_size_is_limited = pytest.mark.skipif(
    gurobi_problem_size_is_limited(),
    reason='requires a full version of Gurobi')
