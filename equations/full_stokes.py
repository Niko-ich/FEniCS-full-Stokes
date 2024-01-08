from dolfin import *
from mshr import *
#from equations.nonlinear_stokes import NonlinearStokes
from equations.stokes import Stokes
from solver.linear import LinearDirect
import numpy as np
#from scipy.interpolate import LinearNDInterpolator

import copy

class FullStokes(Stokes):
    def __init__(self, geometry, fem, nue, n, delta,mu0, f,friction_domain=None,friction_coefficient=0):
        # call the Stokes constructor:
        super().__init__(geometry, fem, nue, f,L=None,friction_domain=friction_domain,friction_coefficient=friction_coefficient)
        # math exponent:
        self.s = 1.0 + 1.0/n
        # parameter to avoid division by zero:
        self.delta = delta
        # parameter to guarantee convergence of newton's method
        self.mu0 = mu0
        
        # Friction is only working for the explicitly given friction
        self.friction_domain = friction_domain
        self.friction_coefficient = friction_coefficient

        # List of functional values
        self.functional_values_list = []
        # vel_of_G is the velocity corresponding to G(v)
        self.vel_of_G = float('NaN')

    def mu(self, u):
        Du = sym(nabla_grad(u))
        return 2*self.nue*(0.5*tr(Du*Du) + self.delta**2)**((self.s-2)/2)
    
    def friction(self,u):
            return 2*self.friction_coefficient*(0.5*dot(u,u)+self.delta**2)**((self.s-2)/2) 

    def abl_mu(self,u):
        Du = sym(nabla_grad(u))
        return (self.s-2)*self.nue*(0.5*tr(Du*Du)+self.delta**2)**((self.s-4)/2)
    
    def abl_friction(self,u):
        return (self.s-2)*self.friction_coefficient*(0.5*dot(u,u)+self.delta**2)**((self.s-4)/2)

    def a_picard(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        a = self.mu(u0)*inner(sym(nabla_grad(u)),nabla_grad(v))*dx \
            -div(v)*p*dx-q*div(u)*dx
        if(self.mu0 != 0.0):
            a = a + self.mu0*inner(nabla_grad(u),nabla_grad(v))*dx
        if(self.friction_domain!=None):
            a = a + self.friction(u0)*inner(u,v)*self.friction_domain
        return a


    def a_newton(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)



        a = self.a_picard()
        # For the case s==2, we have the classical stokes problem.
        if(self.s != 2):
            a = a + self.abl_mu(u0)*inner(sym(nabla_grad(u0)),sym(nabla_grad(u)))\
            *inner(sym(nabla_grad(u0)),nabla_grad(v))*dx

        if(self.friction_domain!=None):
            a = a + self.abl_friction(u0)*inner(u0,u)*inner(u0,v)*self.friction_domain
        return a


    def L_newton(self):
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        L =  self.mu(u0)*inner(sym(nabla_grad(u0)),nabla_grad(v))*dx \
            - inner(self.f,v)*dx-p0*div(v)*dx-q*div(u0)*dx
        if(self.mu0 != 0.0):
            L = L + self.mu0*inner(nabla_grad(u0),nabla_grad(v))*dx
        if(self.friction_domain!=None):
            L = L + self.friction(u0)*inner(u0,v)*self.friction_domain
        return L

    def L_newton_abl_norm(self,uhelp):
        # This term is needed to calculate ||G'(u)G(u)||.
        # U_help corresponds to the velocity v that represents G(u).
        # That is the directional derivative G'(u)uhelp
        u0,p0 = self.U.split()
        (u,p) = TrialFunctions(self.W)
        (v,q) = TestFunctions(self.W)
        L = self.mu(u0)*inner(sym(nabla_grad(uhelp)),nabla_grad(v))*dx
        if(self.mu0 != 0.0):
            L = L + self.mu0*inner(nabla_grad(uhelp),nabla_grad(v))*dx
        if(self.s != 2):
            L = L + self.abl_mu(u0)*inner(sym(nabla_grad(u0)),nabla_grad(uhelp))\
                    *inner(sym(nabla_grad(u0)),nabla_grad(v))*dx
        if(self.friction_domain!=None):
            L = L + self.friction(u0)*inner(uhelp,v)*self.friction_domain \
                +self.abl_friction(u0)*inner(u0,uhelp)*inner(u0,v)*self.friction_domain
        return L

    def functional(self):
        # Minimizing this functional is equivalent to solving the full stokes equations.
        u,p = self.U.split()
        functional_value = assemble((4.0/self.s*self.nue*(0.5*inner(sym(nabla_grad(u)),sym(nabla_grad(u)))+self.delta**2)**(self.s/2)
                         -inner(self.f,u))*dx)
        if(self.mu0 != 0.0):
            functional_value = functional_value + assemble(0.5*self.mu0*inner(sym(nabla_grad(u)),sym(nabla_grad(u)))*dx)
        if(self.friction_domain!=None):
            functional_value = functional_value + assemble((4.0/self.s*self.friction_coefficient*(0.5*inner(u,u)+self.delta**2)**(self.s/2))*self.friction_domain)
        functional_value = functional_value - assemble(div(u)*p*dx)
        return functional_value

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
        functional_deriv_value = assemble(self.mu(uhelp) * inner(sym(nabla_grad(uhelp)), nabla_grad(u)) * dx \
                        - inner(self.f, u) * dx)
        if(self.mu0 != 0.0):
            functional_deriv_value = functional_deriv_value + assemble(self.mu0 * inner(sym(nabla_grad(uhelp)), nabla_grad(u)) * dx)
        if(self.friction_domain!=None):
            functional_deriv_value = functional_deriv_value + assemble(self.friction(uhelp) * inner(uhelp,u)* self.friction_domain)

        functional_deriv_value = functional_deriv_value - assemble((div(u)*phelp-div(uhelp)*p)*dx)
        
        return functional_deriv_value


    def save_data(self,output_file,norm_list,rel_error,rel_local_error,step_sizes,computation_time_iteration,computation_time_step_size):
        # We save data for post-processing.
        super().save_data(output_file)
        # The only additional value that we save is the list of functional values.
        # All other information is saved by the stokes class.
        np.save(str(output_file)+"norm_list", norm_list)
        np.save(str(output_file)+"div_list", self.div_list)
        np.save(str(output_file)+"functional_values_list", self.functional_values_list)
        np.save(str(output_file)+"relative_error",rel_error)
        np.save(str(output_file)+"relative_local_error",rel_local_error)
        np.save(str(output_file)+"step_sizes",step_sizes)
        np.save(str(output_file)+"computation_time_iteration",computation_time_iteration)
        np.save(str(output_file)+"computation_time_step_size",computation_time_step_size)

        if hasattr(self.geometry,"top_grid"):
            np.save(str(output_file)+"top_grid",self.geometry.top_grid)

    def norm(self):
        # Returns the norm value and the velocity field that corresponds to G(v)
        # We create a stokes object and solve the problem with the corresponding right-hand side
        # We need the boundary term with coefficient 1.0
        stokes = Stokes(self.geometry,self.fem,1,1,L=self.L_newton(),friction_domain=self.friction_domain,friction_coefficient=1.0)
        solver = LinearDirect(stokes)
        solver.solve()
        (v2,p2) = stokes.U.split()
        norm_value = assemble(inner(nabla_grad(v2),nabla_grad(v2))*dx)
        if(self.friction_domain!=None):
            norm_value = norm_value + assemble(inner(v2,v2)*self.friction_domain)
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
        stokes = Stokes(self.geometry,self.fem,1,1,L=self.L_newton_abl_norm(self.vel_of_G),friction_domain=self.friction_domain,friction_coefficient=1.0)
        solver = LinearDirect(stokes)
        solver.solve()
        (v2,p2) = stokes.U.split()
        norm_abl_value = np.sqrt(assemble(inner(nabla_grad(v2),nabla_grad(v2))*dx))
        if(self.friction_domain!=None):
            norm_abl_value = np.sqrt(norm_abl_value*norm_abl_value + assemble(inner(v2,v2)*self.friction_domain))
        return norm_abl_value

    def time_step(self,deltaT):
        displacements = []
        top_grid_old = copy.deepcopy(self.geometry.top_grid)
        u,p = self.U.split()
        # ... Is this a problem???
        u.set_allow_extrapolation(True)
        p.set_allow_extrapolation(True)
        print('dimension',self.geometry.dimension)

        bmesh = BoundaryMesh(self.geometry.mesh, "exterior", True)

        # We create the matrices to solve
        left_hand_side_matrix = np.zeros((np.shape(self.geometry.top_grid)[0],np.shape(self.geometry.top_grid)[0]))
        right_hand_side = np.zeros((np.shape(self.geometry.top_grid)[0],1))

        # We save a vector with the indices in bmesh.coordinates and geometry.top.grid
        if(self.geometry.dimension == 2):
            index_vector = np.zeros(np.shape(self.geometry.top_grid),dtype=int)
        counter = 0

        cfl_constant_check = 0.0
        for i in range(0,np.shape(bmesh.coordinates())[0]):
            is_moving = False
            if(self.geometry.dimension == 2):
                for k in range(0,np.shape(self.geometry.top_grid)[0]):
                    if(np.all(self.geometry.top_grid[k] == bmesh.coordinates()[i])):
                        is_moving = True
                        index = k
                        index_vector[counter,0] = k
                        index_vector[counter,1] = i
                        counter = counter + 1


            if(is_moving == True):
                #print('coordinate moving',bmesh.coordinates()[i])
                # We want to calculate dzx.
                # For that purpose, we determine the point left or right to the point that we consider.
                
                # We determine an index that is not the point itself
                if(self.geometry.dimension == 2):
                    index_list = []
                    for k in range(0,np.shape(self.geometry.top_grid)[0]):
                        if(self.geometry.top_grid[k][0] != self.geometry.top_grid[index][0]):
                            if(k != index):
                                index_list.append(k)
                    right_value = np.inf
                    left_value = -np.inf
                    left_index = -1
                    right_index = -1
                    for j in index_list:
                        if(self.geometry.top_grid[j][0] < self.geometry.top_grid[index][0] and self.geometry.top_grid[j][0] > left_value):
                            left_index = j
                            left_value = self.geometry.top_grid[j][0]
                        if(self.geometry.top_grid[j][0] > self.geometry.top_grid[index][0] and self.geometry.top_grid[j][0] < right_value):
                            right_index = j
                            right_value = self.geometry.top_grid[j][0]             


                    # We fill in the matrix values
                    ux = u(bmesh.coordinates()[i])[0]
                    uz = u(bmesh.coordinates()[i])[1]
                    z = bmesh.coordinates()[i][1]


                    Theta = 0.0
                    is_in_bottom = False
                    for k in range(0,np.shape(self.geometry.bottom_grid)[0]):
                        if(np.all(self.geometry.bottom_grid[k] == bmesh.coordinates()[i])):
                            is_in_bottom = True
                    if(is_in_bottom == True):
                        left_hand_side_matrix[index,index] = 1.0
                        right_hand_side[index,0] = z
                    elif(ux > 0):
                        length_dx = self.geometry.top_grid[index][0]-self.geometry.top_grid[left_index][0]
                        if(cfl_constant_check< abs(ux*deltaT/length_dx)):
                            cfl_constant_check = abs(ux*deltaT/length_dx)
                        left_hand_side_matrix[index,index] =  1.0 + Theta*ux*deltaT/length_dx
                        left_hand_side_matrix[index,left_index] = - Theta*ux*deltaT/length_dx
                        right_hand_side[index,0] = z - (1-Theta)*ux*deltaT/length_dx*(self.geometry.top_grid[index][1]-self.geometry.top_grid[left_index][1]) + deltaT * uz
                    else:
                        length_dx = self.geometry.top_grid[right_index][0]-self.geometry.top_grid[index][0]
                        if(cfl_constant_check< abs(ux*deltaT/length_dx)):
                            cfl_constant_check = abs(ux*deltaT/length_dx)
                        left_hand_side_matrix[index,index] =  1.0 - Theta * ux * deltaT/length_dx
                        left_hand_side_matrix[index,right_index] = Theta * ux*deltaT/length_dx
                        right_hand_side[index,0] = z + (1-Theta) * ux*deltaT/length_dx*(self.geometry.top_grid[right_index][1]-self.geometry.top_grid[index][1]) + deltaT * uz

  
        print('cfl constant',cfl_constant_check)
        # We save the old top domain to check, if we have a stationary solution
        old_top = copy.deepcopy(self.geometry.top_grid)
        print('sum surface velocity',np.sum(bmesh.coordinates()))
 
        solution_vector = np.linalg.solve(left_hand_side_matrix,right_hand_side)


        for k in range(0,np.shape(self.geometry.top_grid)[0]):
            self.geometry.top_grid[k][1] =  solution_vector[k,0]
            bmesh.coordinates()[index_vector[k,1]][1] = solution_vector[index_vector[k,0],0]



        print('diff',np.linalg.norm(np.subtract(old_top,self.geometry.top_grid)))
        if(np.linalg.norm(np.subtract(old_top,self.geometry.top_grid))<0.1):
            print('reached stationary point')
        massing = Constant(1.0)
        u,p = self.U.split()
        massing = project(massing, FunctionSpace(self.geometry.mesh,'Lagrange',1))
        print('old div(v)*div(v): ',assemble(dot(div(u),div(u))*dx))
        print('old div(v)',assemble(div(u)*dx))
        print('mass old: ',assemble(massing*dx))
        ALE.move(self.geometry.mesh,bmesh)
        print('new functional value',self.functional())
        massing = project(massing, FunctionSpace(self.geometry.mesh,'Lagrange',1))
        u,p = self.U.split()
        print('new div(v)*div(v): ',assemble(dot(div(u),div(u))*dx))
        print('new div(v)',assemble(div(u)*dx))
        print('mass new: ',assemble(massing*dx))
        # Fixes the error that would happen after some iterations
        self.geometry.mesh.bounding_box_tree().build(self.geometry.mesh)