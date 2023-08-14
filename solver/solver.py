from dolfin import *
import numpy as np

class Solver:
    def __init__(self, pde):
        # The solver needs as attributes the pde object with all its attributes.
        self.equation = pde

