import sys
sys.path.append("../")

from decimal import Decimal

from elements.taylor_hood import *
from equations.full_stokes import *
#from geometries.arola import *
from geometries.arola_e2 import *

from solver.picard import Picard
from solver.newton import Newton
from step_size_control.exact import *
from step_size_control.armijo import *
from step_size_control.constant_step import *

dim = 2

Domain = ArolaE2(length_x=5000,resolution=140,dimension=2,num_more_grid=3,time_dependent=False)

ArolaE2.visualize(Domain)


Elements = TaylorHood(Domain)

n = 3
delta = 10**(-12)
mu0 = 0.0
B = 0.5*(1.0e-16)**(-1.0/3.0)
realistic_factor = 10**6 # factor that together with B has a typical size of Dv
g = 9.81
rho = 910
f = g*rho*Constant((0,-1))

equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Exact(min_term='functional',right=4.0,left=0.0,threshold=0.0)
solver = Newton(equation,line_search,max_iter=40,epsi=1e-3,calc_norm=True,output_file='ArolaE2/Stationary/Newton_exact')
solution_Newton_exact = solver.solve()


equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Armijo(min_term='functional',gamma=10e-10,min_step=0.5)
solver = Newton(equation,line_search,max_iter=40,epsi=1e-3,calc_norm=True,output_file='ArolaE2/Stationary/Newton_armijo')
solution_Newton = solver.solve()

equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = ConstantStep()
solver = Picard(equation,line_search,max_iter=40,epsi=1e-3,calc_norm=True,output_file='ArolaE2/Stationary/Picard')
solution_Picard = solver.solve()


equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Exact(min_term='functional',right=4.0,left=0.0,threshold=0.0)
solver = Picard(equation,line_search,max_iter=40,epsi=1e-3,calc_norm=True,output_file='ArolaE2/Stationary/Picard_exact')
solution_Picard_exact = solver.solve()
