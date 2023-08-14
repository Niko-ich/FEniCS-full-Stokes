from dolfin import *
from mshr import *
#from equations.nonlinear_stokes import NonlinearStokes
from equations.stokes import Stokes
from solver.linear import LinearDirect
import numpy as np

class FullStokes(Stokes):
    def __init__(self, geometry, fem, nue, n, delta,mu0, f):
        # call the Stokes constructor:
        super().__init__(geometry, fem, nue, f)
        # math exponent:
        self.s = 1.0 + 1.0/n
        # parameter to avoid division by zero:
        self.delta = delta
        # parameter to guarantee convergence of newton's method
        self.mu0 = mu0

        # List of functional values
        self.functional_values_list = []
        # vel_of_G is the velocity corresponding to G(v)
        self.vel_of_G = float('NaN')

    def mu(self, u):
        Du = sym(nabla_grad(u))
        return 2*self.nue*(0.5*tr(Du*Du) + self.delta**2)**((self.s-2)/2)

    def abl_mu(self,u):
        Du = sym(nabla_grad(u))
        return (self.s-2)*self.nue*(0.5*tr(Du*Du)+self.delta**2)**((self.s-4)/2)

    def a_picard(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        a = self.mu0*inner(nabla_grad(u),nabla_grad(v))*dx \
            +self.mu(u0)*inner(sym(nabla_grad(u)),nabla_grad(v))*dx \
            -div(v)*p*dx-q*div(u)*dx
        return a

    def a_newton(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)

        a = self.a_picard()\
            +self.abl_mu(u0)*inner(sym(nabla_grad(u0)),sym(nabla_grad(u)))\
            *inner(sym(nabla_grad(u0)),nabla_grad(v))*dx
        return a


    def L_newton(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        L =  self.mu0*inner(nabla_grad(u0),nabla_grad(v))*dx \
            + self.mu(u0)*inner(sym(nabla_grad(u0)),nabla_grad(v))*dx \
            - inner(self.f,v)*dx-p0*div(v)*dx-q*div(u0)*dx
        return L

    def L_newton_abl_norm(self,uhelp):
        # This term is needed to calculate ||G'(u)G(u)||.
        # U_help corresponds to the velocity v that represents G(u).
        # That is the directional derivative G'(u)uhelp
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        L = self.mu0*inner(nabla_grad(uhelp),nabla_grad(v))*dx\
            +self.mu(u0)*inner(sym(nabla_grad(uhelp)),nabla_grad(v))*dx\
            +self.abl_mu(u0)*inner(sym(nabla_grad(u0)),nabla_grad(uhelp))\
            *inner(sym(nabla_grad(u0)),nabla_grad(v))*dx
        return L

    def functional(self):
        # Minimizing this functional is equivalent to solving the full stokes equations.
        u,p = self.U.split()
        return assemble((4.0/self.s*self.nue*(0.5*inner(sym(nabla_grad(u)),sym(nabla_grad(u)))+self.delta**2)**(self.s/2)
                         +0.5*self.mu0*inner(sym(nabla_grad(u)),sym(nabla_grad(u)))-inner(self.f,u))*dx)

    def functional_derivative(self,U,t):
        # We calculate the derivative of the reduced functional J_h
        # J(u)= int_{\Omega}(0.5 |Du|^2+delta^2)^{p/2}
        # with J_h(t)=J(u+tw)=int_{Omega}(0.5 |D(u+tw)|^2+delta^2)^{p/2}+mu0*(nabla(u+tw),nabla(u+tw))-(rho g,u+tw)
        # The derivative follows with the chain rule:
        # J_h'(t)=int_{Omega}p/2 (0.5 |D(u+tw)|^2+delta^2)^{(p-2)/2} D(u+tw):Dw+mu0+2*mu0*(nabla(u+tw),nabla(u))
        # - (rho g,w)
        # In other words: We calculate <G(u+tw),w>
        # Here u is the direction and self.u is the position
        # U: direction
        # t: step size
        u,p = U.split()
        U_help = Function(self.W)
        U_help.assign(self.U+t*U)
        uhelp,phelp = U_help.split()
        return assemble(self.mu0 * inner(sym(nabla_grad(uhelp)), nabla_grad(u)) * dx \
                        + self.mu(uhelp) * inner(sym(nabla_grad(uhelp)), nabla_grad(u)) * dx \
                        - inner(self.f, u) * dx)


    def save_data(self,output_file,norm_list,rel_error,rel_local_error):
        # We save data for post-processing.
        super().save_data(output_file)
        # The only additional value that we save is the list of functional values.
        # All other information is saved by the stokes class.
        np.save(str(output_file)+"norm_list", norm_list)
        np.save(str(output_file)+"div_list", self.div_list)
        np.save(str(output_file)+"functional_values_list", self.functional_values_list)
        np.save(str(output_file)+"relative_error",rel_error)
        np.save(str(output_file)+"relative_local_error",rel_local_error)

    def norm(self):
        # Returns the norm value and the velocity field that corresponds to G(v)
        # We create a stokes object and solve the problem with the corresponding right-hand side
        stokes = Stokes(self.geometry,self.fem,1,1,L=self.L_newton())
        solver = LinearDirect(stokes)
        solver.solve()
        (v2,p2) = stokes.U.split()
        norm_value = assemble(inner(nabla_grad(v2),nabla_grad(v2))*dx)
        self.vel_of_G = v2
        return norm_value

    def norm_derivative(self,U=None,t=None):
    	# Second and third argument are dummy arguments 
    	# that are needed for the functional_derivative.
    	# But we want to have the same input arguments as in functional_derivative.
        # We calculate ||G'(u)G(u)||.
        # Here U_help is U_help=RG(u) with the Riesz-Isomporphism R.
        #print('uhelp',u_help)
        #print('other vel_g',self.vel_of_G)
        stokes = Stokes(self.geometry,self.fem,1,1,L=self.L_newton_abl_norm(self.vel_of_G))
        solver = LinearDirect(stokes)
        solver.solve()
        (v2,p2) = stokes.U.split()
        norm_abl_value = np.sqrt(assemble(inner(nabla_grad(v2),nabla_grad(v2))*dx))
        return norm_abl_value

