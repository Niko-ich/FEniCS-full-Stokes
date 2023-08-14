from dolfin import *
import numpy as np
from solver.solver import Solver
class LinearDirect(Solver):
    def __init__(self, pde):
        # call constructor from superclass:
        super().__init__(pde)

    def solve(self,inner_solver="mumps"):
        # That is the linear direct solver for linear problems
        solve(self.equation.a == self.equation.L, self.equation.U, self.equation.bc,solver_parameters={"linear_solver": inner_solver})
