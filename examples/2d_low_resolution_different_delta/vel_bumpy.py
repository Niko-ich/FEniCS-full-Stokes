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


velocities_Newton_Armijo_1e_minus3 = np.load('Newton_with_Armijo_delta0.001velocities.npy')
velocities_Newton_Armijo_1e_minus4 = np.load('Newton_with_Armijo_delta0.0001velocities.npy')
velocities_Newton_Armijo_1e_minus5 = np.load('Newton_with_Armijo_delta1e-12velocities.npy')
velocities_Picard_Constant = np.load('Picard_constant_delta1e-12velocities.npy')
coor = np.load('Picard_constant_delta1e-12coordinates.npy')

# We create a list for the indices of the relevant coordinates
# and a list of the relevant coordinates
(relevant_coord,x2) = relevant_coordinates(coor)
#(relevant_coord_unrefined,x2_unrefined) = relevant_coordinates(coor_unrefined)
# Next, we obtain the surface velocity calculates as in ISMIP-HOM

#vel_norm_Newton_exact = obtain_velocity(velocities_Newton_exact,relevant_coord)
vel_norm_Newton_Armijo_1e_minus3 = obtain_velocity(velocities_Newton_Armijo_1e_minus3,relevant_coord)
vel_norm_Newton_Armijo_1e_minus4 = obtain_velocity(velocities_Newton_Armijo_1e_minus4,relevant_coord)
vel_norm_Newton_Armijo_1e_minus5 = obtain_velocity(velocities_Newton_Armijo_1e_minus4,relevant_coord)

vel_norm_Picard_Constant = obtain_velocity(velocities_Picard_Constant,relevant_coord)

plt.rcParams['figure.figsize'] = (8,6)


font=14
width = 2
plt.ylim(10.0,12.5)




plt.plot(x2/5000.0,vel_norm_Newton_Armijo_1e_minus3,label='Newton Armijo with $\delta=10^{-3}$',color='#1b9e77',linewidth=width,marker='x')
plt.plot(x2/5000.0,vel_norm_Newton_Armijo_1e_minus4,label='Newton Armijo with $\delta=10^{-4}$',color='#1b9e77',linewidth=width,marker='s')
plt.plot(x2/5000.0,vel_norm_Newton_Armijo_1e_minus5,label='Newton Armijo with $\delta=10^{-12}$',color='#1b9e77',linewidth=width,marker='+')
plt.plot(x2/5000.0,vel_norm_Picard_Constant,label='Picard',color='#7570b3',linewidth=width)


plt.rcParams["text.usetex"] = True
plt.xlabel('x position normalized',fontsize=font)
plt.ylabel('Velocity in m a$^{-1}$',fontsize=font)
plt.rcParams["text.usetex"] = False
plt.legend()
plt.grid()
plt.show()



