from ..cplex import big_cplex_problem, small_cplex_problem


def test_big_cplex_problem():
    """
    .. test:: Test a large CPLEX problem
       :links: PROBLEM_SIZE_LIMITS.CPLEX
    """
    big_cplex_problem()


def test_small_cplex_problem():
    small_cplex_problem()
