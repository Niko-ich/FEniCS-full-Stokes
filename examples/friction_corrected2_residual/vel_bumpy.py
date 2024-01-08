import matplotlib.pyplot as plt
import numpy as np

def relevant_coordinates(coor):
	relevant_coord = list()
	for k in range(0,np.shape(coor)[0]):
		if(abs(coor[k][1]-1000.0)<1.0 and coor[k][0]>-1.0 and coor[k][0]<5001.0):
			relevant_coord.append(k)
	x2 = np.zeros(len(relevant_coord))
	counter = 0
	for k in relevant_coord:
		x2[counter] = coor[k,0]
		counter = counter + 1
	return (relevant_coord,x2)

def obtain_velocity(vel,relevant_coord):
	# vel: velocities
	# coor: coordinates
	vel_norm = np.zeros(len(relevant_coord))
	alpha = 10.0*np.pi/180.0
	counter = 0
	for k in relevant_coord:
		vel_norm[counter] = np.sqrt((vel[k][0]*np.cos(alpha))**2-(vel[k][1]*np.sin(alpha))**2)
		counter = counter+1
	return vel_norm

velocities_Picard_Constant_high_coefficient = np.load('150x30Picard_Reference_friction_coeff_high_res10000000.0velocities.npy')
velocities_Picard_Constant_low_coefficient = np.load('150x30Picard_Reference_friction_coeff_high_res1000.0velocities.npy')

coor = np.load('150x30Picard_Reference_friction_coeff_high_res10000000.0coordinates.npy')

font = 14
plt.rcParams['figure.figsize'] = (8,6)

# we only visualize every nth element
n = 64

plt.quiver(coor[::n,0],coor[::n,1],velocities_Picard_Constant_low_coefficient[::n,0],velocities_Picard_Constant_low_coefficient[::n,1])
plt.xlim((0,5000))
plt.grid()
plt.xlabel('Coordinate in x-direction in m',fontsize=font)
plt.ylabel('Coordinate in y-direction in m',fontsize=font)
plt.show()


plt.quiver(coor[::n,0],coor[::n,1],velocities_Picard_Constant_high_coefficient[::n,0],velocities_Picard_Constant_high_coefficient[::n,1])
plt.xlim((0,5000))
plt.grid()
plt.xlabel('Coordinate in x-direction in m',fontsize=font)
plt.ylabel('Coordinate in y-direction in m',fontsize=font)
plt.show()


