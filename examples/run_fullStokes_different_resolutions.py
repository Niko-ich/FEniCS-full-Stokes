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
dim=2
num_x = 24
num_z = 2
for j in range(0,5):
	Domain = BumpyBed(k=3,resolution_x_direction=num_x,resolution_z_direction=num_z,local_refine=False)
	
	hmax = Domain.mesh.hmax()
	print('hmax',hmax)
	print('num cells',Domain.mesh.num_cells())
	# We visualize our domain
	np.save("2d_run_final_different_resolutions/hmax_with_resx"+str(num_x), hmax)
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

	n= 3 # Exponent for nonlinear term with p=1+1/n
	delta = 10**(-12) # term for differentiation.
	mu0 = 0.0 # term formaly needed for getting solution in H^1. However, we set it to zero.
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
	solver = Picard(equation,line_search,max_iter=80,calc_norm=False,output_file='2d_run_final_different_resolutions/reference_calculation_without_mu0_resx'+str(num_x))
	solution_picard_ref = solver.solve()


	print('Newton with armijo')
	# Newton with armijo
	equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
	# We solve the problem with a velocity field that we introduce in nue
	initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
	initial_solve = LinearDirect(initial_stokes)
	initial_solve.solve()
	equation.U = initial_stokes.U
	line_search = Armijo(min_term='functional',gamma=10**(-10))
	solver = Newton(equation,line_search,max_iter=40,calc_norm=True,output_file='2d_run_final_different_resolutions/Newton_with_Armijo_without_mu0_resx'+str(num_x),ref_solution=solution_picard_ref)
	solution_Newton_armijo = solver.solve()

	# Setting the resolution for the next experiment
	num_x = num_x * 2
	num_z = num_z * 2



