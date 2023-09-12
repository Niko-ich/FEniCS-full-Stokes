import sys
from step_size_control.step_size_control import *
class Exact(StepSizeControl):

    def __init__(self,min_term='norm',min_step=0.0,right=4.0,left=0.0,max_bisection_iterations=25):

        super().__init__(min_term,min_step)
        self.right = right
        self.left = left

        self.max_bisection_iterations = max_bisection_iterations

    def calc_step_size(self, U, solver):
        # This calculates the exact step size.
        # The exact step size is only calculated if a functional is given.
        # Otherwise, the armijo step size is used.
        minimization_term = getattr(solver.equation,self.min_term)

        if (self.min_term == 'norm'):
            sys.exit('Not possible to use the exact step size. Formula does not work. '
                     'No exact step size computable with the algorithm.')
        # Depending on the derivative we decide for the left respectively right interval.
        # This calculates the exact step size as we have convex functional
        # and we can use for example Theorem 6.1 in Schoelkopf Learning with Kernels.
        # We need the negative of the direction, because we solved G'(v)u=G(v) instead of G'(v)u=-G(v)
        U.assign(-1.0 * U)
        right = self.right
        left = self.left
        for k in range(0, self.max_bisection_iterations):
            if (solver.equation.functional_derivative(U, (left + right) / 2.0) > 0):
                right = (left + right) / 2.0
            else:
                left = (left + right) / 2.0
        t = (left + right) / 2.0
        # We use a minimal step size
        if(t<self.min_step):
            t = self.min_step
        print('step size is',t)
        solver.equation.U.assign(solver.equation.U + t * U)
        solver.step_sizes.append(t)
        return solver.equation.U
