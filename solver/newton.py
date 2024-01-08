from solver.nonlinear_iterative import NonlinearIterative
import numpy as np
from dolfin import *
import time 
class Newton(NonlinearIterative):

    def __init__(self,pde,step_size_control,max_iter=15,epsi=1e-10,inner_solver="mumps",calc_norm=False,output_file = 'output',ref_solution=None):
        super().__init__(pde,step_size_control,max_iter,epsi,inner_solver,calc_norm,output_file,ref_solution)

    def solve(self):
        # This is the Newton solver
        # Calculates diagnostic variables like norm etc.
        self.calculate_diagnostic_variables()
        # We calculate the residual norm to check if ||G(u_0)||/||G(u_k)|| is small enough.
        # Then we stop.
        #old_residual = self.equation.norm()
        
        for iter in range(self.max_iter):
            print('iter',iter)
            print('functional value: ',self.equation.functional())
            start = time.time()
            # We calculate the solution for Newton's method
            a = self.equation.a_newton()
            L = self.equation.L_newton()
            U_Newton = Function(self.equation.W)
            solve(a == L, U_Newton, self.equation.bc, solver_parameters={"linear_solver": self.inner_solver})


            # We save the old velocity field to check later, if a stopping criteria is fulfilled
            U_old = Function(self.equation.W)
            U_old.assign(self.equation.U)

            # We update the field self.equation.U with the new value
            # We pass information like the equation and the minimization term
            start_step_size_calc = time.time()
            self.step_size_control.calc_step_size(U_Newton,self)
            end_step_size_calc = time.time()
            self.compuation_time_step_size.append(end_step_size_calc-start_step_size_calc)
            # We perform diagnostic calculations.
            # Please modify this method in nonlinear_iterative.py corresponding to your needs
            end = time.time()
            self.computation_time_iteration.append(end-start)
            # We do not consider the time for calculating diagnostics.
            print('time one iteration',end-start)
            self.diagnostic_stop()
            u_old, p_old = U_old.split()
            u,p = self.equation.U.split()
            rel_error = 2.0*assemble(inner(u_old-u,u_old-u)*dx)/(assemble(inner(u_old,u_old)*dx)+assemble(inner(u,u)*dx))
            print('rel_error_squared',rel_error)
            #new_residual = self.equation.norm()
            #if(new_residual/old_residual < self.epsi**2):
            #    break

        self.equation.save_data(self.output_file,self.norm_list,self.rel_error,self.rel_local_error,self.step_sizes,self.computation_time_iteration,self.compuation_time_step_size)
        return self.equation.U
