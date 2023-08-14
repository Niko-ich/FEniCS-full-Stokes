from dolfin import *
from mshr import *
from elements.element import Element

class TaylorHood(Element):
    def __init__(self, geometry):
        P2 = VectorElement("Lagrange", geometry.mesh.ufl_cell(), 2)
        P1 = FiniteElement("Lagrange", geometry.mesh.ufl_cell(), 1)
        self.space = FunctionSpace(geometry.mesh, P2 * P1)
