from dolfin import *
import numpy as np
from geometries.bumpy_bed import BumpyBed

class BumpyBed3D(BumpyBed):

	def __init__(self,k=3,length_x=5000,resolution_x_direction=2,resolution_z_direction=11,local_refine=False):
		# k is the number of copies to the left and right.
		# L is the length_x of the domain in x-direction in meters.
		# resolution_x_direction is the resolution in x-dircetion.
		# resolution_z_direction is the resolution z-direction.
		super().__init__(dimension=3)
		self.k = k
		self.length_x = length_x
		self.resolution_x_direction = resolution_x_direction
		self.resolution_z_direction = resolution_z_direction
		self.local_refine = local_refine

		def bottom_3d(x, on_boundary):
			return on_boundary and x[2]< 500.0*sin(2.0*np.pi/self.length_x*x[0])*sin(2.0*np.pi/self.length_x*x[1])+100.0

		def front_3d(x,on_boundary):
			return on_boundary and x[1]> (self.k+1.0)*self.length_x - 10.0

		def back_3d(x,on_boundary):
			return on_boundary and x[1]< -self.k * self.length_x + 10.0

		def left(x, on_boundary):
			return on_boundary and x[0] < -self.k * self.length_x + 10.0

		def right(x, on_boundary):
			return on_boundary and x[0] > (self.k + 1.0) * self.length_x - 10.0

		self.boundaries = [bottom_3d,front_3d,back_3d,left,right]

		# Now we create the domain
		pre_domain = BoxMesh(Point(-self.k * self.length_x, -self.k * self.length_x, 0),
								 Point((self.k + 1) * self.length_x, (self.k + 1) * self.length_x, 1000), self.resolution_x_direction,
								 self.resolution_x_direction, self.resolution_z_direction)

		coor = pre_domain.coordinates()

		# We do a local refinement if it is wanted
		# That only works for self.resolution_x_direction=98
		coor = self.local_refine_domain(coor)

		# We remap for the bottom
		for j in range(0, np.shape(coor)[0]):
			coor[j, :] = self.remap_3d(coor[j, :])

		# We can save the mesh now
		self.mesh = pre_domain

	def set_boundary_conditions(self,V):
		# We set the boundary conditions corresponding to the ISMIP-HOM A experiment
		noslip = Constant((0.0, 0.0, 0.0))
		bc_bottom = DirichletBC(V, noslip, self.boundaries[0])
		bc_left = DirichletBC(V, noslip, self.boundaries[1])
		bc_right = DirichletBC(V, noslip, self.boundaries[2])
		bc_front = DirichletBC(V, noslip, self.boundaries[3])
		bc_back = DirichletBC(V, noslip, self.boundaries[4])

		return [bc_bottom, bc_left, bc_right, bc_front, bc_back]


	def remap_3d(self,x):
		x[2] = 1000.0 - x[2] * (1000.0 - 500.0 * np.sin(2 * np.pi / self.length_x * x[0]) * np.sin(
			2 * np.pi / self.length_x * x[1])) / 1000.0
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
		# function for refining mesh locally in area [0,L] and corasing it in [-3L,0] and [L,3L].
		l = 7 * self.length_x
		# The calculation is the same as in bumpy_bed.py. Additionally, we have the third dimension.
		num_spaces = 98.0
		spaces_com_first = 11.0
		spaces_com_second = 24.0
		spaces_com_third = 39.0
		spaces_com_fourth = 59.0
		spaces_com_fivth = 74.0
		spaces_com_sixth = 87.0
		y = np.zeros(3)
		# We do not change the vertical component
		y[2] = x[2]
		# The component in x1-direction is changed identically to the 2d dimensional case
		y[0] = super().local_refinement(x)[0]

		if (self.dimension == 3):
			if (-3 * self.length_x + spaces_com_sixth / num_spaces * l < x[1]):
				y[1] = 3 * self.length_x + (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_sixth) / (
							num_spaces - spaces_com_sixth) * self.length_x

			if (-3 * self.length_x + spaces_com_fivth / num_spaces * l < x[
				1] <= -3 * self.length_x + spaces_com_sixth / num_spaces * l):
				y[1] = 2 * self.length_x + (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_fivth) / (
							spaces_com_sixth - spaces_com_fivth) * self.length_x

			if (-3 * self.length_x + spaces_com_fourth / num_spaces * l < x[
				1] <= -3 * self.length_x + spaces_com_fivth / num_spaces * l):
				y[1] = self.length_x + (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_fourth) / (
							spaces_com_fivth - spaces_com_fourth) * self.length_x

			if (-3 * self.length_x + spaces_com_third / num_spaces * l < x[
				1] <= -3 * self.length_x + spaces_com_fourth / num_spaces * l):
				y[1] = (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_third) / (
							spaces_com_fourth - spaces_com_third) * self.length_x

			if (-3 * self.length_x + spaces_com_second / num_spaces * l < x[
				1] <= -3 * self.length_x + spaces_com_third / num_spaces * l):
				y[1] = -self.length_x + (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_second) / (
							spaces_com_third - spaces_com_second) * self.length_x

			if (-3 * self.length_x + spaces_com_first / num_spaces * l < x[
				1] <= -3 * self.length_x + spaces_com_second / num_spaces * l):
				y[1] = -2 * self.length_x + (num_spaces / l * (3 * self.length_x + x[1]) - spaces_com_first) / (
							spaces_com_second - spaces_com_first) * self.length_x

			if (x[1] <= -3 * self.length_x + spaces_com_first / num_spaces * l):

				y[1] = -3 * self.length_x + (num_spaces * self.length_x) / (spaces_com_first * l) * (
							x[1] + 3.0 * self.length_x)

		return y

