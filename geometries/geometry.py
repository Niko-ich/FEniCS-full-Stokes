import matplotlib.pyplot as plt
import numpy as np
from dolfin import *

class Geometry():

	def __init__(self,dimension):
		self.dimension = dimension


	def visualize(self):
		coor = self.mesh.coordinates()
		if(self.dimension==3):
			x = coor[:,0]
			y = coor[:,1]
			z = coor[:,2]
			# We do a control plot
			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			scatter = ax.scatter(x,y,z)
			plt.legend(*scatter.legend_elements(),title='grid points')
			plt.show()

			# We create a surface plot
			x_list = []
			y_list = []
			z_list = []
			print('len(z)',len(z))
			# First, we find the surface
			surface = max(z)
			print('surface_value',surface)
			for i in range(0,len(x)):
				if(z[i]>=surface-0.001):
					x_list.append(x[i])
					y_list.append(y[i])
					z_list.append(z[i])

			fig = plt.figure()
			ax = fig.add_subplot(projection='3d')
			scatter = ax.scatter(x_list, y_list, z_list)
			plt.legend(*scatter.legend_elements(), title='grid points surface')
			plt.show()

		if (self.dimension == 2):
			print(self.mesh.coordinates())
			coor_list = self.mesh.coordinates()
			plot(self.mesh)
			plt.show()

