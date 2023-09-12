import sys
sys.path.append("../")

from elements.taylor_hood import *
from equations.full_stokes import *
from geometries.bumpy_bed import *
# for 3d simulation
#from geometries.bumpy_bed_3d import *
from solver.picard import Picard
from solver.newton import Newton
from step_size_control.exact import *
from step_size_control.armijo import *
from step_size_control.constant_step import *

import matplotlib.pyplot as plt
import numpy as np

# We chose the dimension of the domain. Either dim=2 or dim=3.
dim = 2
Domain = BumpyBed(k=3,resolution_x_direction=140,resolution_z_direction=10,local_refine=False)
# in 3d
#dim = 3
#Domain = BumpyBed3D(k=3,resolution_x_direction=6,resolution_z_direction=3,local_refine=False)
# local_refine is created for a reduced number of grid points in the ISMIP-HOM experiments
# at the copies of the glaciers. It only works for resolution_x_direction=98.

# We visualize our domain
BumpyBed.visualize(Domain)
#BumpyBed3D.visualize(Domain)
# We chose the elements
Elements = TaylorHood(Domain)
# We set constants for the problem
alpha = 0.5*np.pi/180.0
g = 9.81
rho = 910
if(dim==2):
	f = g * rho * Constant((np.sin(alpha),-np.cos(alpha)))
if(dim==3):
	f = g * rho * Constant((np.sin(alpha),0,-np.cos(alpha)))

# We set the constants
n = 3 # Exponent for nonlinear term with p=1+1/n
delta = 10**(-12) # term for differentiation.
mu0 = 10**(-17)	# term for getting solution in H^1.
B = 0.5*(1.0e-16)**(-1.0/3.0) # B term. Factor infront of nonlinear term.
realistic_factor = 10**6 # factor that together with B has a typical size of Dv

# We choose the p-Stokes/FullStokes equations.
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
# We obtain a first guess by solving the Stokes problem.
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
# We solve the linear stokes problem in one step.
initial_solve = LinearDirect(initial_stokes)
# We call the solver
initial_solve.solve()
# We save the solution of the stokes problem to our equation.
equation.U = initial_stokes.U
# We chose the line search
line_search = Armijo(min_term='functional',gamma=10**(-10))
# min_term = 'norm' is also possible. 
# That calculates the residual norm by solving a Stokes problem. 
# Alternatives are
#line_search = Exact(min_term='functional',right=4.0,left=0.0)
#line_search = ConstantStep(step_size=1.0)
# We choose the solver
#solver = Picard(equation,line_search,max_iter=3,calc_norm=True, output_file='test')
# Alternatively, we can choose Newton's method for the solver
solver = Newton(equation,line_search,max_iter=3,calc_norm=True, output_file='test')
# Picard and Newton have the optional argument ref_solution.
# That can be used to calculate the relative error and a local relative error.
# Finaly, we call the solver
solution_picard_ref = solver.solve()
