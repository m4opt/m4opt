from ..gurobi import big_gurobi_problem, small_gurobi_problem


def test_big_gurobi_problem():
    """
    .. test:: Test a large Gurobi problem
       :links: PROBLEM_SIZE_LIMITS.GUROBI
    """
    big_gurobi_problem()


def test_small_gurobi_problem():
    small_gurobi_problem()
