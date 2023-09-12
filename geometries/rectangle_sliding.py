from dolfin import *
from mshr import *
import numpy as np
from geometries.geometry import Geometry

class RectangleSliding(Geometry):
	def __init__(self,length_x=5000,length_z=1000,resolution_x_direction=20,resolution_z_direction=11,dimension=2):
    	# k is the number of copies to the left and right.
    	# L is the length_x of the domain in x-direction in meters.
    	# resolution_x_direction is the resolution in x-dircetion.
    	# resolution_z_direction is the resolution z-direction.
		super().__init__(dimension)
		self.dimension = dimension
		self.length_x = length_x
		self.length_z = length_z
		self.resolution_x_direction = resolution_x_direction
		self.resolution_z_direction = resolution_z_direction

		# We define the boundaries
		class bottom(SubDomain):
			def inside(self,x, on_boundary):
				return on_boundary and x[1] < 100.0 

		class left(SubDomain):
			def inside(self,x, on_boundary):
				return on_boundary and x[0] < 50.0
		
		class right(SubDomain):
			def inside(self,x,on_boundary):
				return on_boundary and x[0] > 5000.0-50.0

		# Next we create the domain
		# We create a rectangle mesh
		pre_domain = RectangleMesh(Point(0,0),Point(self.length_x,1000),self.resolution_x_direction,self.resolution_z_direction)
		# We need the coordinates to shift the coordinates corresponding to the bumpy bed
		coor = pre_domain.coordinates()


		# We can save the mesh now
		self.mesh = pre_domain


		self.boundaries = MeshFunction('size_t',self.mesh,self.mesh.topology().dim()-1)
		#self.boundaries = [bottom_sliding, left,right,top]
		#bottom_sliding().mark(markers, 1) # marking to 1 while all other facets are 0
		left().mark(self.boundaries,1)
		right().mark(self.boundaries,2)
		bottom().mark(self.boundaries,0)

		self.ds =Measure('ds',domain = self.mesh , subdomain_data = self.boundaries)





	


	def set_boundary_conditions(self,V):
		# We set the boundary conditions corresponding to the ISMIP-HOM B experiment
		
		# We define the boundaries
		class bottom(SubDomain):
			def inside(self,x, on_boundary):
				return on_boundary and x[1] < 100.0 
		
		class left(SubDomain):
			def inside(self,x, on_boundary):
				return on_boundary and x[0] < 50.0
		
		class right(SubDomain):
			def inside(self,x,on_boundary):
				return on_boundary and x[0] > 5000.0-50.0
		
		noslip_1d = Constant(0.0)
		noslip = Constant((0.0,0.0))
		bc_bottom = DirichletBC(V.sub(1),noslip_1d,bottom())
		bc_left = DirichletBC(V,noslip,left())
		bc_right = DirichletBC(V,noslip,right())
		#bc_left = DirichletBC(V, noslip, self.boundaries[1])
		#bc_bottom_sliding = DirichletBC(V,noslip,self.boundaries[1])
		#bc_right = DirichletBC(V, noslip, self.boundaries[2])
		#bc_top = DirichletBC(V,noslip, self.boundaries[3])
		#return bc_bottom_sliding
		return [bc_bottom,bc_left,bc_right]
	

