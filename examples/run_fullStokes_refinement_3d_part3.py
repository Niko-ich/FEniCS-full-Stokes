import sys
sys.path.append("../")

from elements.taylor_hood import *
from equations.full_stokes import *
from geometries.bumpy_bed_3d import *
from solver.picard import Picard
from solver.newton import Newton
from step_size_control.exact import *
from step_size_control.armijo import *
from step_size_control.constant_step import *

import matplotlib.pyplot as plt
import numpy as np
import time


#print('precision',np.finfo(float).eps)

# Creating channel
#domain = ChannelCylinder()
dim=3
#Domain = BumpyBed(k=3,resolution_x_direction=98,resolution_z_direction=10,local_refine=True)
#Domain = BumpyBed(k=3,resolution_x_direction=140,resolution_z_direction=10,local_refine=False)
#Domain = BumpyBed(k=3,resolution_x_direction=70,resolution_z_direction=10,local_refine=False)
Domain = BumpyBed3D(k=3,resolution_x_direction=98,resolution_z_direction=10,local_refine=True)
#Domain = BumpyBed3D(k=3,resolution_x_direction=36,resolution_z_direction=10,local_refine=False)
#Domain.create_domain(local_refine=True)


print('num cells',Domain.mesh.num_cells())
#print('backends',list_linear_algebra_backends())
#print('used',linear_algebra_backends())
#print('list solvers',list_linear_solver_methods())
#print('list preconditioners',list_krylov_solver_preconditioners())

# We visualize our domain
#BumpyBed3D.visualize(Domain)

start = time.time()

Elements = TaylorHood(Domain)


alpha=0.5*np.pi/180.0
g=9.81
rho=910
if(dim==2):
	f = g*rho*Constant((np.sin(alpha),-np.cos(alpha)))
if(dim==3):
	f = g*rho*Constant((np.sin(alpha),0,-np.cos(alpha)))



# Newton with armijo step sizes
'''
Equation = FullStokes(Domain,Elements,0.5*(1.0e-16)**(-1.0/3.0),3,10**(-12),10**(-17),f)
Initial_Stokes = Stokes(Domain,Elements,0.5*(1.0e-16)**(-1.0/3.0)*10**6,f)
Initial_Solve = LinearDirect(Initial_Stokes)
Initial_Solve.solve()
Equation.U = Initial_Stokes.U
Stepping = StepSizeControl(gamma=10**(-10))
solver = Picard(Equation,Stepping,max_iter=40,calc_norm=True,step_method='constant',min_term='functional',output_file='test')
solution_Picard = solver.solve()
'''

#File mesh_file("u.pvd");
#u_field >> u;
n= 3 # Exponent for nonlinear term with p=1+1/n
delta = 10**(-12) # term for differentiation.
mu0 = 0.0
B = 0.5*(1.0e-16)**(-1.0/3.0) # B term. Factor infront of nonlinear term.
realistic_factor = 10**6 # factor that together with B has a typical size of Dv


print('reference calculation')
# reference solution
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = ConstantStep()
solver = Picard(equation,line_search,max_iter=80,calc_norm=False,output_file='/gxfs_home/cau/sunip606/fenics_simulation_v.1.3.0/examples/simulation_high_resolution_corrected/part3/reference_solution')
solution_picard_ref = solver.solve()


'''
print('Newton with armijo')
# Newton with armijo
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Armijo(min_term='functional',gamma=10**(-10))
solver = Newton(equation,line_search,max_iter=40,calc_norm=True,output_file='2d_run_final_corrected_relvel/Newton_with_Armijo',ref_solution=solution_picard_ref)
solution_Newton_armijo = solver.solve()


print('Newton with armijo min_step')
# Newton with armijo min_step
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Armijo(min_term='functional',gamma=10**(-10),min_step=0.5)
solver = Newton(equation,line_search,max_iter=40,calc_norm=True,
				output_file='2d_run_final_corrected_relvel/Newton_with_Armijo_min_step',ref_solution=solution_picard_ref)
solution_Newton_armijo_min_step = solver.solve()

print('Newton with exact')
# Newton with exact
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Exact(min_term='functional',right=4.0,left=0.0)
#solver = Newton(equation,line_search,max_iter=40,calc_norm=True,step_method='exact',min_term='functional',
#				output_file='exact_newton_test',ref_solution=solution_picard_ref)
solver = Newton(equation,line_search,max_iter=40,calc_norm=True,
				output_file='2d_run_final_corrected_relvel/Newton_with_exact',ref_solution=solution_picard_ref)

solution_Newton_exact = solver.solve()


print('Picard constant')
# Picard
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = ConstantStep()
solver = Picard(equation,line_search,max_iter=40,calc_norm=True,output_file='2d_run_final_corrected_relvel/Picard_constant',ref_solution=solution_picard_ref)
solution_picard = solver.solve()
'''
print('Picard exact step size')
# Picard exact step size
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search =Exact(min_term='functional',right=4.0,left=0.0)
solver = Picard(equation,line_search,max_iter=40,calc_norm=True,
output_file='/gxfs_home/cau/sunip606/fenics_simulation_v.1.3.0/examples/simulation_high_resolution_corrected/part3/Picard_exact',ref_solution=solution_picard_ref)
solution_picard_exact = solver.solve()

print('Picard armijo step size')
# Picard exact step size
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We solve the problem with a velocity field that we introduce in nue
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search =Armijo(min_term='functional',gamma=10**(-10))
solver = Picard(equation,line_search,max_iter=40,calc_norm=True,
output_file='/gxfs_home/cau/sunip606/fenics_simulation_v.1.3.0/examples/simulation_high_resolution_corrected/part3/Picard_armijo',ref_solution=solution_picard_ref)
solution_picard_armijo = solver.solve()

end = time.time()
print('elapsed time',end-start)

