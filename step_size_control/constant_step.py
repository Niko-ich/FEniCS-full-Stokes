from step_size_control.step_size_control import *

class ConstantStep(StepSizeControl):

    def __init__(self, step_size=1.0):
        # We do not really need the upper class for the Constant step size.
        super().__init__(min_term=None, min_step=step_size)
        self.step_size = step_size

    def calc_step_size(self, U,solver):
        # U: direction
        # solver: information about the solver
        print('step size',self.step_size)
        solver.equation.U.assign(solver.equation.U - self.step_size * U)
        return solver.equation.U