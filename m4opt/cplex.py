from docplex.mp.model import Model


def big_cplex_problem():
    m = Model()
    m.binary_var_list(10000)
    m.solve()


def small_cplex_problem():
    m = Model()
    m.binary_var_list(1000)
    m.solve()
