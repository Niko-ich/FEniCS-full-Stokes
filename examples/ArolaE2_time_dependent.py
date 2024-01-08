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

Domain = ArolaE2(length_x=5000,resolution=140,dimension=2,num_more_grid=3,time_dependent=True)

#ArolaE2.visualize(Domain)


Elements = TaylorHood(Domain)

n = 3
delta = 10**(-12)
mu0 = 0.0
B = 0.5*(1.0e-16)**(-1.0/3.0)
realistic_factor = 10**6 # factor that together with B has a typical size of Dv
g = 9.81
rho = 910
f = g*rho*Constant((0,-1))

k = 2
deltaT = 1e0/(2**k)
num_years = 30
'''
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Exact(min_term='functional',right=4.0,left=0.0,threshold=0.0)
solver = Newton(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Newton_exact/Newton_exactiter0')
for i in range(0,num_years*2**k):
    print('outer_iter',i)
    solution_Newton_exact = solver.solve()
    equation.time_step(deltaT)
    solver = Newton(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Newton_exact/Newton_exactiter'+str(i))
'''

equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Armijo(min_term='functional',gamma=10e-10,min_step=0.5)
solver = Newton(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Newton_armijo/Newton_armijoiter0')
for i in range(0,num_years*2**k):
    print('outer_iter',i)
    solution_Newton_armijo = solver.solve()
    equation.time_step(deltaT)
    solver = Newton(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Newton_armijo/Newton_armijoiter'+str(i))
'''
equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = ConstantStep()
solver = Picard(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Picard/Picarditer0')
for i in range(0,num_years*2**k):
    print('outer_iter',i)
    solution_Picard = solver.solve()
    equation.time_step(deltaT)
    solver = Picard(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Picard/Picarditer'+str(i))


equation = FullStokes(Domain,Elements,B,n,delta,mu0,f)
initial_stokes = Stokes(Domain,Elements,B*realistic_factor,f)
initial_solve = LinearDirect(initial_stokes)
initial_solve.solve()
equation.U = initial_stokes.U
line_search = Exact(min_term='functional',right=4.0,left=0.0,threshold=0.0)
solver = Picard(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Picard_exact/Picard_exactiter0')
for i in range(0,num_years*2**k):
    print('outer_iter',i)
    solution_Picard_exact = solver.solve()
    equation.time_step(deltaT)
    solver = Picard(equation,line_search,max_iter=50,epsi=10**(-3),calc_norm=True,output_file='ArolaE2/Instationary/Picard_exact/Picard_exactiter'+str(i))
'''
