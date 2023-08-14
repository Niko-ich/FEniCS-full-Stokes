from dolfin import *
from mshr import *
import numpy as np
from geometries.geometry import Geometry

class BumpyBed(Geometry):
	def __init__(self,k=3,length_x=5000,resolution_x_direction=20,resolution_z_direction=11,local_refine=False,dimension=2):
    	# k is the number of copies to the left and right.
    	# L is the length_x of the domain in x-direction in meters.
    	# resolution_x_direction is the resolution in x-dircetion.
    	# resolution_z_direction is the resolution z-direction.
		super().__init__(dimension)
		self.k = k
		self.length_x = length_x
		self.resolution_x_direction = resolution_x_direction
		self.resolution_z_direction = resolution_z_direction
		self.local_refine = local_refine

		# We define the boundaries
		def bottom(x, on_boundary):
			return on_boundary and x[1] < 500.0 * sin(2.0 * np.pi / self.length_x * x[0]) + 100.0

		def left(x, on_boundary):
			return on_boundary and x[0] < -self.k * self.length_x + 10.0

		def right(x, on_boundary):
			return on_boundary and x[0] > (self.k + 1.0) * self.length_x - 10.0


		self.boundaries = [bottom, left, right]

		# Next we create the domain
		# We create a rectangle mesh
		pre_domain = RectangleMesh(Point(-self.k*self.length_x,0),Point((self.k+1)*self.length_x,1000),self.resolution_x_direction,self.resolution_z_direction)
		# We need the coordinates to shift the coordinates corresponding to the bumpy bed
		coor = pre_domain.coordinates()

		# We do a local refinement if it is wanted
		# That only works for self.resolution_x_direction=98
		coor = self.local_refine_domain(coor)
        # We remap for the bottom
		for j in range(0,np.shape(coor)[0]):
			coor[j,:] = self.remap(coor[j,:])

		# We can save the mesh now
		self.mesh = pre_domain

	def set_boundary_conditions(self,V):
		# We set the boundary conditions corresponding to the ISMIP-HOM B experiment
		noslip = Constant((0.0,0.0))
		bc_bottom = DirichletBC(V, noslip, self.boundaries[0])
		bc_left = DirichletBC(V, noslip, self.boundaries[1])
		bc_right = DirichletBC(V, noslip, self.boundaries[2])
		return [bc_bottom,bc_left,bc_right]
	def remap(self,x):
		# function for remapping to the bottom
		x[1] = 1000.0 - x[1] * (1000.0 - 500.0 * np.sin(2 * np.pi / self.length_x * x[0])) / 1000.0

		return x

	def local_refine_domain(self,grid_points):
		coor_copy = np.zeros((np.shape(grid_points)[0],np.shape(grid_points)[1]))
		if(self.local_refine==True and self.resolution_x_direction==98):
			# We transform the copy
			for j in range(0,np.shape(grid_points)[0]):
				coor_copy[j,:] = self.local_refinement(grid_points[j,:])
			# We set the values of the original
			for j in range(0,np.shape(grid_points)[0]):
				grid_points[j,:] = coor_copy[j,:]
		return grid_points

	def local_refinement(self,x):
		# function for refining mesh locally in area [0,L] and coarsening it in [-3L,0] and [L,3L].
		l = 7 * self.length_x
		# We define the number of spaces (between two nodes is one space).
		num_spaces = 98.0
		# We also define the number of spaces we have from the end of one copy to the next.
		# That means our first copy of the glacier hast spaces_com_first+1 grid points
		# and spaces_com_first spaces.
		# The second copy of the glaicher has spaces_com_second-spaces_com_first+1 grid points
		# and spaces_com_second-spaces_com_first spaces.
		spaces_com_first = 11.0
		spaces_com_second = 24.0
		spaces_com_third = 39.0
		spaces_com_fourth = 59.0
		spaces_com_fifth = 74.0
		spaces_com_sixth = 87.0

		y = np.zeros(2)
		y[1] = x[1]
		if (-3 * self.length_x + spaces_com_sixth / num_spaces * l < x[0]):
			y[0] = 3 * self.length_x + (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_sixth) / (
						num_spaces - spaces_com_sixth) * self.length_x

		if (-3 * self.length_x + spaces_com_fifth / num_spaces * l < x[
			0] <= -3 * self.length_x + spaces_com_sixth / num_spaces * l):
			y[0] = 2 * self.length_x + (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_fifth) / (
						spaces_com_sixth - spaces_com_fifth) * self.length_x

		if (-3 * self.length_x + spaces_com_fourth / num_spaces * l < x[
			0] <= -3 * self.length_x + spaces_com_fifth / num_spaces * l):
			y[0] = self.length_x + (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_fourth) / (
						spaces_com_fifth - spaces_com_fourth) * self.length_x

		if (-3 * self.length_x + spaces_com_third / num_spaces * l < x[
			0] <= -3 * self.length_x + spaces_com_fourth / num_spaces * l):
			y[0] = (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_third) / (
						spaces_com_fourth - spaces_com_third) * self.length_x

		if (-3 * self.length_x + spaces_com_second / num_spaces * l < x[
			0] <= -3 * self.length_x + spaces_com_third / num_spaces * l):
			y[0] = -self.length_x + (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_second) / (
						spaces_com_third - spaces_com_second) * self.length_x

		if (-3 * self.length_x + spaces_com_first / num_spaces * l < x[
			0] <= -3 * self.length_x + spaces_com_second / num_spaces * l):
			y[0] = -2 * self.length_x + (num_spaces / l * (3 * self.length_x + x[0]) - spaces_com_first) / (
						spaces_com_second - spaces_com_first) * self.length_x

		if (x[0] <= -3 * self.length_x + spaces_com_first / num_spaces * l):

			y[0] = -3 * self.length_x + (num_spaces * self.length_x) / (spaces_com_first * l) * (x[0] + 3.0 * self.length_x)

		return y

