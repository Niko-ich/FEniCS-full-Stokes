from dolfin import *
import numpy as np

class PDE:
    def __init__(self, geometry, fem):
        self.geometry = geometry
        self.fem = fem
        # define function space:
        self.W = fem.get_space()


