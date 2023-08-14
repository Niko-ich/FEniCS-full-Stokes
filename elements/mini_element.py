from dolfin import *
from mshr import *
from elements.element import Element

class MiniElement(Element):
    def __init__(self, geometry):
        P1 = FiniteElement("Lagrange", geometry.mesh.ufl_cell(), 1)
        B  = FiniteElement("Bubble", geometry.mesh.ufl_cell(), 3)
        self.space = FunctionSpace(geometry.mesh, (P1+B) * P1)
