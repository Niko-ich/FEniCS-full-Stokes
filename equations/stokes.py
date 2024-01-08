from dolfin import *
from mshr import *
import matplotlib.pyplot as plt
import numpy as np
from equations.pde import PDE

class Stokes(PDE):
    def __init__(self, geometry, fem, nue, f,L=None,friction_domain=None,friction_coefficient=0):
        super().__init__(geometry,fem)
        (u, p) = TrialFunctions(self.W)
        (v, q) = TestFunctions(self.W)
        # define equation:
        self.nue = nue
        self.a = self.nue*inner(grad(u), grad(v))*dx \
                 + div(v)*p*dx + q*div(u)*dx
        
        # Additional friction term
        if(friction_domain!=None):
            self.a = self.a + friction_coefficient*inner(u,v)*friction_domain

        # PSPG stabilization if necessary and defined in fem space:
        if hasattr(self.fem, 'stab'):
            self.a += self.stab_PSPG()*inner(grad(p), grad(q))*dx
        if(L==None):
            self.f = f
            self.L = inner(self.f, v)*dx
        else:
            self.L = L
        self.U = Function(self.W)
        # set boundary conditions
        # parameter is velocity space
        self.bc = self.geometry.set_boundary_conditions(self.W.sub(0))

        # We can check, if our solutions are divergence-free.
        self.div_list = []

    # PSPG global stabilization parameter by Tezduyar:
    def stab_PSPG(self):
        Reloc = self.geometry.maxU*self.geometry.mesh.hmin()/(2*self.nue)
        Reloc = min(Reloc/3,1)
        return self.geometry.mesh.hmin()*Reloc/(2*self.geometry.maxU)



    def obtain_velocity_fields(self,U):
        # We calculate the velocity and pressure fields.
        coordinates = self.geometry.mesh.coordinates()
        velocities = np.zeros((np.shape(coordinates)[0], self.geometry.dimension))
        pressures = np.zeros((np.shape(coordinates)[0], 1))
        u, p = U.split()


        for i in range(0,np.shape(coordinates)[0]):
            pressures[i,0] = p(coordinates[i][:])
            for j in range(0,self.geometry.dimension):
                velocities[i,j] = u(coordinates[i][:])[j]

        return velocities, pressures

    def save_data(self, output_file):
        # We save velocity and pressure field with the corresponding coordinates
        coordinates = self.geometry.mesh.coordinates()
        velocities = np.zeros((np.shape(coordinates)[0], self.geometry.dimension))
        pressures = np.zeros((np.shape(coordinates)[0], 1))
        u, p = self.U.split()
        # ... Is this a problem???
        u.set_allow_extrapolation(True)
        p.set_allow_extrapolation(True)
        for i in range(0,np.shape(coordinates)[0]):
            pressures[i,0] = p(coordinates[i][:])
            for j in range(0,self.geometry.dimension):
                velocities[i,j] = u(coordinates[i][:])[j]

        np.save(str(output_file)+"velocities",velocities)
        np.save(str(output_file)+"pressures", pressures)
        np.save(str(output_file)+"coordinates", coordinates)
        File(str(output_file)+"u.pvd")<<u
        File(str(output_file)+"p.pvd")<<p
