from . import skip_if_gurobi_problem_size_is_limited
from .. import big_gurobi_problem, small_gurobi_problem


@skip_if_gurobi_problem_size_is_limited
def test_big_gurobi_problem():
    big_gurobi_problem()


def test_small_gurobi_problem():
    small_gurobi_problem()