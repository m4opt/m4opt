from gurobipy import Model


def big_gurobi_problem():
    m = Model()
    m.addMVar(10000)
    m.optimize()


def small_gurobi_problem():
    m = Model()
    m.addMVar(1000)
    m.optimize()
