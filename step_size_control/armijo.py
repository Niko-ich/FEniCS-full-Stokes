from dolfin import *
import numpy as np
from step_size_control.step_size_control import *

class Armijo(StepSizeControl):

    def __init__(self,min_term='norm',min_step=10**(-6),threshold = 0.0,gamma=10**(-8)):

        super().__init__(min_term,min_step)
        self.gamma = gamma
        self.threshold = threshold

    def calc_step_size(self,U,solver):
        # This calculates the classical armijo step size.

        # We get the minimization term from the attributes
        minimization_term = getattr(solver.equation,self.min_term)
        # The derivative for the minimization term has the same name and additionally _derivative
        minimization_derivative = getattr(solver.equation,self.min_term+str('_derivative'))
        # We get the old value of the minimization term.
        minimization_term_old = getattr(solver,self.min_term+str('_old'))
        # We obtain the old minimization value, if we calculated it
        if(np.isnan(minimization_term_old)):
            min_term_old = minimization_term()
        else:
            min_term_old = minimization_term_old
        gradient = minimization_derivative(U,1.0)
        alpha = 1.0

        solver.equation.U.assign(solver.equation.U - alpha * U)

        min_term_new = minimization_term()
        while (min_term_new > min_term_old - self.gamma * alpha * gradient and alpha > self.min_step):
            alpha = 0.5 * alpha
            solver.equation.U.assign(solver.equation.U + alpha * U)
            min_term_new = minimization_term()
            #if(alpha < self.threshold):
            #    solver.equation.U.assign(solver.equation.U + (alpha-1.0) * U)
            #    alpha = 1.0
            #    break

        # We set the new value for the old minimization term
        setattr(solver,self.min_term+str('_old'),min_term_new)

        print('alpha from '+str(self.min_term), alpha)
        solver.step_sizes.append(alpha)
        print('min_term relative difference', "{:.8E}".format((min_term_old - min_term_new) / min_term_old))
        return solver.equation.U
