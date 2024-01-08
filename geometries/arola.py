from dolfin import *
from mshr import *
import numpy as np
from geometries.geometry import Geometry
import pandas as pd
import os

class Arola(Geometry):
	def __init__(self,length_x=5000,resolution=16,dimension=2,num_more_grid=3,time_dependent=False):
		super().__init__(dimension)
		self.length_x = length_x
		self.resolution = resolution
		self.initial_domain = pd.read_csv("arola.txt",sep="	",header=None)
		self.initial_domain[1][:] = self.initial_domain[1][:]
		self.initial_domain[2][:] = self.initial_domain[2][:]
		self.top_grid = []
		self.bottom_grid = []
		
		# First variant for creating the mesh
		domain_vertices = []		
		for k in range(0,len(self.initial_domain[1][:])):
			domain_vertices.append(Point(self.initial_domain[0][k],self.initial_domain[1][k]))
			self.bottom_grid.append([self.initial_domain[0][k],self.initial_domain[1][k]])
			if(k<len(self.initial_domain[2][:])-1):
				for i in range(1,num_more_grid):
					x_right = self.initial_domain[0][k+1]
					x_left = self.initial_domain[0][k]
					z_right = self.initial_domain[1][k+1]
					z_left = self.initial_domain[1][k]
					domain_vertices.append(Point(x_left+i/num_more_grid *(x_right-x_left),
								  	z_left+i/num_more_grid*(z_right-z_left)))
					self.bottom_grid.append([x_left+i/num_more_grid*(x_right-x_left),
						   			z_left+i/num_more_grid*(z_right-z_left)])
		print('bottom',self.bottom_grid)

		# We add a grid point to have a variable hight for the time-dependent problem
		if(time_dependent == True):
			domain_vertices.append(Point(5000.0,2505.0))
		for k in range(0,len(self.initial_domain[2][:])):
			if(k>0):
				domain_vertices.append(Point(self.initial_domain[0][len(self.initial_domain[2][:])-1-k],self.initial_domain[2][len(self.initial_domain[2][:])-1-k]))
			# We add grid points at the top linear interpolated between the known coordinates
			if(k<len(self.initial_domain[2][:])-1):
				for i in range(1,num_more_grid):
					x_right = self.initial_domain[0][len(self.initial_domain[2][:])-1-k]
					x_left = self.initial_domain[0][len(self.initial_domain[2][:])-2-k]
					z_right = self.initial_domain[2][len(self.initial_domain[2][:])-1-k]
					z_left = self.initial_domain[2][len(self.initial_domain[2][:])-2-k]
					domain_vertices.append(Point(x_right+i/num_more_grid *(x_left-x_right),
								  	z_right+i/num_more_grid*(z_left-z_right)))
		domain = Polygon(domain_vertices)
		self.mesh = generate_mesh(domain,resolution)
		# We add all boundary point to the top grid, except the left and the right grid point.

		print('initial_domain',self.initial_domain)

		bmesh = BoundaryMesh(self.mesh, "exterior",True)
		print('bmesh_coordinates',bmesh.coordinates())
		for i in range(0,np.shape(bmesh.coordinates())[0]):
			k = 0
			while(self.initial_domain[0][k] < bmesh.coordinates()[i][0]):
				k = k + 1
			# We interpolate between the known grid points at the top
			if(k>0):
				length = self.initial_domain[0][k] - self.initial_domain[0][k-1]
				l1 = 1-(self.initial_domain[0][k] - bmesh.coordinates()[i][0]) / length
				l2 = 1-(bmesh.coordinates()[i][0] - self.initial_domain[0][k-1]) / length
				if (bmesh.coordinates()[i][1] > l1*self.initial_domain[2][k]+l2*self.initial_domain[2][k-1]-0.2):
					# We do not add the [5000.0, 0.0] bottom point for the time dependent simulation
					if(time_dependent== True):
						if(np.all(bmesh.coordinates()[i] == [5000.0,2500.0])):
							print('did not add grid point at [5000.0,2500.0] from list of top grid')
						else:
							self.top_grid.append(bmesh.coordinates()[i])
					else:
						self.top_grid.append(bmesh.coordinates()[i])
			else:
				self.top_grid.append(bmesh.coordinates()[i])


		# Convert to list
		self.top_grid = [list_entry.tolist() for list_entry in self.top_grid]
		print('top_grid_list',self.top_grid)

		def bottom(x, on_boundary):
			k = 0
			while(self.initial_domain[0][k] < x[0]):
				k = k + 1
			# We interpolate between the known grid points at the top
			if(k>0):
				length = self.initial_domain[0][k] - self.initial_domain[0][k-1]
				l1 = 1-(self.initial_domain[0][k] - x[0]) / length
				l2 = 1-(x[0] - self.initial_domain[0][k-1]) / length
				return on_boundary and x[1] < l1*self.initial_domain[1][k]+l2*self.initial_domain[1][k-1]+0.2
			else:
				return on_boundary
			
		self.boundaries = [bottom]

	def set_boundary_conditions(self,V):
		# We set the boundary conditions corresponding to the ISMIP-HOM experiment E1
		noslip = Constant((0.0,0.0))
		slip = Constant(0.0)
		bc_bottom = DirichletBC(V,noslip,self.boundaries[0])
		return [bc_bottom]
	
		
