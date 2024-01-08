import sys
sys.path.append("../")

from elements.taylor_hood import *
from equations.full_stokes import *
from geometries.rectangle_sliding import *
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
dim=2
#Domain = BumpyBed(k=3,resolution_x_direction=98,resolution_z_direction=10,local_refine=True)
Domain = RectangleSliding(resolution_x_direction=150,resolution_z_direction=30)
#Domain = BumpyBed(k=3,resolution_x_direction=70,resolution_z_direction=10,local_refine=False)
#Domain = BumpyBed3D(k=3,resolution_x_direction=98,resolution_z_direction=10,local_refine=True)
#Domain = BumpyBed3D(k=3,resolution_x_direction=36,resolution_z_direction=10,local_refine=False)
#Domain.create_domain(local_refine=True)


print('num cells',Domain.mesh.num_cells())
#print('backends',list_linear_algebra_backends())
#print('used',linear_algebra_backends())
#print('list solvers',list_linear_solver_methods())
#print('list preconditioners',list_krylov_solver_preconditioners())

# We visualize our domain
#RectangleSliding.visualize(Domain)

start = time.time()

Elements = TaylorHood(Domain)


alpha=0.5*np.pi/180.0
g=9.81
rho=910
if(dim==2):
	f = g*rho*Constant((np.sin(alpha),-np.cos(alpha)))


#File mesh_file("u.pvd");
#u_field >> u;
n= 3 # Exponent for nonlinear term with p=1+1/n
delta = 10**(-12) # term for differentiation.
mu0= 10**(-17)	# term for getting solution in H^1.
B = 0.5*(1.0e-16)**(-1.0/3.0) # B term. Factor infront of nonlinear term.
realistic_factor = 10**6 # factor that together with B has a typical size of Dv

friction_coefficients = [1e3,1e7]


for friction_coefficient in friction_coefficients:

	print('reference calculation')
	# reference solution
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f,friction_domain=Domain.ds(0),friction_coefficient=friction_coefficient)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search = ConstantStep()
	solver = Picard(equation,line_search,max_iter=40,calc_norm=True,output_file='friction_other_start_value/150x30Picard_Reference_friction_coeff_high_res'+str(friction_coefficient))
	solution_picard_ref = solver.solve()

	print('Newton with armijo')
	# Newton with armijo
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f,friction_domain=Domain.ds(0),friction_coefficient=friction_coefficient)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search = Armijo(min_term='functional',gamma=10**(-15))
	solver = Newton(equation,line_search,max_iter=40,calc_norm=True,output_file='friction_other_start_value/150x30Newton_with_Armijo_friction_coeff_high_res'+str(friction_coefficient))
	solution_Newton_armijo = solver.solve()

	print('Newton with armijo min_step')
	# Newton with armijo min_step
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f,friction_domain=Domain.ds(0),friction_coefficient=friction_coefficient)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search = Armijo(min_term='functional',gamma=10**(-10),min_step=0.5)
	solver = Newton(equation,line_search,max_iter=40,calc_norm=True,
				output_file='friction_other_start_value/150x30Newton_with_Armijo_min_step_friction_coeff_high_res'+str(friction_coefficient))
	solution_Newton_armijo_min_step = solver.solve()

	print('Newton with exact')
	# Newton with exact
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f,friction_domain=Domain.ds(0),friction_coefficient=friction_coefficient)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search = Exact(min_term='functional',right=4.0,left=0.0)
	solver = Newton(equation,line_search,max_iter=40,calc_norm=True,
				output_file='friction_other_start_value/150x30Newton_with_exact_friction_coeff_high_res'+str(friction_coefficient))

	solution_Newton_exact = solver.solve()


	print('Picard exact step size')
	# Picard exact step size
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f,friction_domain=Domain.ds(0),friction_coefficient=friction_coefficient)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search =Exact(min_term='functional',right=4.0,left=0.0)
	solver = Picard(equation,line_search,max_iter=40,calc_norm=True,
	output_file='friction_other_start_value/150x30Picard_exact_friction_coeff_high_res'+str(friction_coefficient))
	solution_picard_exact = solver.solve()

end = time.time()
print('elapsed time',end-start)
