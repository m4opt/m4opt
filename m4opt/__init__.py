from gurobipy import Model

from ._astropy_init import *   # noqa

__all__ = []


def big_gurobi_problem():
    m = Model()
    m.addMVar(10000)
    m.optimize()


def small_gurobi_problem():
    m = Model()
    m.addMVar(1000)
    m.optimize()
