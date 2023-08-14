from dolfin import *
import numpy as np
import sys

class StepSizeControl:

    def __init__(self, min_term='norm',min_step=10 ** (-6)):
        # We initalize constants for armjo and exact line search.
        # There are no common attributes between the methods
        self.min_term = min_term
        self.min_step = min_step

 
