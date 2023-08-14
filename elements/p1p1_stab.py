from dolfin import *
from mshr import *
from elements.element import Element

class P1P1Stab(Element):
    def __init__(self, geometry):
        V = VectorElement("Lagrange", geometry.mesh.ufl_cell(), 1)
        P = FiniteElement("Lagrange", geometry.mesh.ufl_cell(), 1)
        self.space = FunctionSpace(geometry.mesh, V * P)
        # default value for PSPG stabilization:
        self.stab = 0.025*geometry.mesh.hmin()**2
