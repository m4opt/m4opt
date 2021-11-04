# Model interface for pyomo general solver layer

import numpy as np

# potential wrapper library for solver interfaces
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

solvers = ['cbc', 'gurobi', 'cplex']

# will extend from appropriate library
class Model():

    def __init__(solver=None):
        """
        solver : string, name of solver to pass to pyomo
        """
        self.set_solver(solver)
        self.model = pyo.ConcreteModel()
        self.sets = []
        self.params = []
        self.variables = []
        return

    def set_solver(self, solver):
        """
        solver : string, name of solver to pass to pyomo
        """
        self.solver = solver
    
    def add_set(self, name, **kwargs):
        """
        Set : index over which to aggregate/sum 
        redundant : could instead do self.model.name = pyo.Set(**kwargs)
        """
        setattr(self.model, name, pyo.Set(**kwargs))
        self.sets.append(name)
        
    def add_param(self, name, **kwargs):
        """
        Param : 
        name : string, parameter name
        **kwargs : keywords for pyomo.environ.Param

        illustrative example
        redundant : could instead do self.model.name = pyo.Param(**kwargs) 
        """
        setattr(self.model, name, pyo.Param(**kwargs))
        self.params.append(name)

    def add_variable(self, name, **kwargs):
        """
        Variable : 
        name : string, parameter name
        **kwargs : keywords for pyomo.environ.Param

        illustrative example
        redundant : could instead do self.model.name = pyo.Var(**kwargs) 
        """
        setattr(self.model, name, pyo.Var(**kwargs))
        self.variables.append(name)

    def add_constraint(self, m4opt_Constraint):
        """
        Take m4opt.Constraint.constraint and convert it into pyomo format
        """

        pass

    def add_objective(self, m4opt_Objective):
        """
        Take m4opt.Constraint.constraint and convert it into pyomo format
        """

        pass
        
    def solve(self):
        opt = pyo.SolverFactory(self.solver)
        opt.solve(self.model)
        
