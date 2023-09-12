from solver.solver import Solver
import numpy as np
from fenics import *

class NonlinearIterative(Solver):
    def __init__(self, pde,step_size_control,max_iter=15,epsi=1e-10,inner_solver="mumps",calc_norm = False,output_file='output',ref_solution= None):
        super().__init__(pde)
        # We give an Object with the information regarding the step size control
        self.step_size_control = step_size_control
        # We set the initial norm
        self.norm_old = float('NaN')
        # We set the initial functional value
        self.functional_old = float('NaN')

        # We save the list of norms
        self.norm_list = []
        # We create a list for relative errors compared to a reference solution
        self.rel_error = []
        # We create a list for local relative errors compared to a reference solution
        self.rel_local_error = []
        # We create a list for the step sizes
        self.step_sizes = []

        # We create two attributes to measure computation time of each iteration
        # and computation time of the step sizes
        self.computation_time_iteration = []
        self.compuation_time_step_size = []

        # Reference solution
        self.reference_solution = ref_solution

        # maximum number of iterations
        self.max_iter = max_iter
        # Stopping criteria for differences between iterations
        self.epsi = epsi
        # inner solver for the linear system
        self.inner_solver = inner_solver
        # Decides, if we calculate the residual norm
        self.calc_norm = calc_norm
        # Output file
        self.output_file = output_file

    def calculate_diagnostic_variables(self):
        if (self.calc_norm == True):
            if (np.isnan(self.norm_old)):
                self.norm_old = self.equation.norm()
                print('residual norm', "{:.8E}".format(self.norm_old))
            elif (self.step_size_control == 'armijo'):
                if (self.min_term == 'norm'):
                    print('residual norm', "{:.8E}".format(self.norm_old))
                else:
                    self.norm_old = self.equation.norm()
                    print('residual norm', "{:.8E}".format(self.norm_old))
            else:
                self.norm_old = self.equation.norm()
                print('residual norm', "{:.8E}".format(self.norm_old))
        self.norm_list.append(self.norm_old)
        if (hasattr(self.equation, 'functional')):
            self.equation.functional_values_list.append(self.equation.functional())
        u, p = self.equation.U.split()
        self.equation.div_list.append(assemble(inner(div(u), div(u)) * dx))

    def diagnostic_stop(self):
        self.calculate_diagnostic_variables()
        if(self.reference_solution is not None):
            u_ref,p_ref = self.reference_solution.split()
            u,p = self.equation.U.split()
            # We calculate the relative error
            rel_temp = assemble(inner(u-u_ref,u-u_ref)*dx)/assemble(inner(u_ref,u_ref)*dx)
            self.rel_error.append(rel_temp)
            # We also calculate the relative local error
            # For that purpose we use an estimate for the velocity: 1mm/year=0.001m/year
            min_vel = Constant(1.0/1000.0)
            def Max(a,b):
                return (a+b+abs(a-b))/2.0
            rel_local_temp = assemble(inner(u-u_ref,u-u_ref)/(Max(inner(u_ref,u_ref),min_vel*min_vel))*dx)
            print('rel_local_temp',rel_local_temp)
            self.rel_local_error.append(rel_local_temp)
