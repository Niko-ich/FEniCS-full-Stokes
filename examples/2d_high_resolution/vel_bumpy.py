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
	alpha = 0.5*np.pi/180.0
	counter = 0
	for k in relevant_coord:
		vel_norm[counter] = np.sqrt((vel[k][0]*np.cos(alpha))**2-(vel[k][1]*np.sin(alpha))**2)
		counter = counter+1
	return vel_norm


velocities_Newton_exact = np.load('Newton_with_exactvelocities.npy')
velocities_Newton_Armijo = np.load('Newton_with_Armijovelocities.npy')
velocities_Newton_Armijo_min_step = np.load('Newton_with_Armijo_min_stepvelocities.npy')
velocities_Picard_exact = np.load('Picard_exactvelocities.npy')
velocities_Picard_Constant = np.load('Picard_constantvelocities.npy')
coor = np.load('Picard_constantcoordinates.npy')

# We create a list for the indices of the relevant coordinates
# and a list of the relevant coordinates
(relevant_coord,x2) = relevant_coordinates(coor)
# Next, we obtain the surface velocity calculates as in ISMIP-HOM
vel_norm_Newton_exact = obtain_velocity(velocities_Newton_exact,relevant_coord)
vel_norm_Newton_Armijo = obtain_velocity(velocities_Newton_Armijo,relevant_coord)
vel_norm_Newton_Armijo_min_step = obtain_velocity(velocities_Newton_Armijo_min_step,relevant_coord)
vel_norm_Picard_exact = obtain_velocity(velocities_Picard_exact,relevant_coord)
vel_norm_Picard_Constant = obtain_velocity(velocities_Picard_Constant,relevant_coord)

font = 14
plt.rcParams['figure.figsize'] = (8,6)


plt.plot(x2,vel_norm_Newton_exact,label='Newton exact')
plt.plot(x2,vel_norm_Newton_Armijo,label='Newton Armijo')
plt.plot(x2,vel_norm_Newton_Armijo_min_step,label='Newton Armijo min step')
plt.plot(x2,vel_norm_Picard_exact,label='Picard exact')
plt.plot(x2,vel_norm_Picard_Constant,label='Picard classic')
plt.xlabel('Coordinate in x-direction in m',fontsize=font)
plt.ylim(4.0,14.0)
plt.ylabel('Velocity in m/a',fontsize=font)
plt.legend()
plt.grid()
plt.show()


# we only visualize every nth element
n = 12

plt.quiver(coor[::n,0],coor[::n,1],velocities_Picard_Constant[::n,0],velocities_Picard_Constant[::n,1])
plt.xlim((0,5000))
plt.grid()
plt.xlabel('Coordinate in x-direction in m',fontsize=font)
plt.ylabel('Coordinate in y-direction in m',fontsize=font)
plt.show()




