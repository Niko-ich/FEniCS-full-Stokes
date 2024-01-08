import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def relevant_coordinates(coor,top_grid):
	relevant_coord = []
	for k in range(0,np.shape(coor)[0]):
		for i in range(0,np.shape(top_grid)[0]):
			if(np.all(coor[k]==top_grid[i])):
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
	alpha = 0.0
	counter = 0
	for k in relevant_coord:
		vel_norm[counter] = np.sqrt(vel[k][0]**2)
		counter = counter+1
	return vel_norm

def interpolate_to_new_grid(dataset,vel_index,new_size):
	# velocity_index: the index with the velocity is dependent on the dataset.
	# new_size: size of the new field.
	vel_field = np.zeros(new_size+1)
	for k in range(0,new_size+1):
		index = 0
		while(dataset[0][index]<= k*1.0/new_size and index<np.shape(dataset)[0]-1):
			index = index + 1
		if(index>0 and k*1.0/new_size<dataset[0][index]):
			index = index - 1
			# Now we have dataset[0][index] <= k*1.0/new_size < dataset[0][index+1]
			weight_left = 1-abs(dataset[0][index]-k*1.0/new_size)/(dataset[0][index+1]-dataset[0][index])
			weight_right = 1-abs(dataset[0][index+1]-k*1.0/new_size)/(dataset[0][index+1]-dataset[0][index])
			vel_field[k] = weight_left*dataset[vel_index][index]+weight_right*dataset[vel_index][index+1]
		elif(dataset[0][index]==1):
			vel_field[k] = dataset[vel_index][index]
		elif(index ==np.shape(dataset)[0]-1):
			vel_field[k] = float("NaN")
		else:
			vel_field[k] = float("NaN")
		vel_field[k] = np.abs(vel_field[k])
			
	return vel_field


velocities_newton_exact = np.load('Newton_exactvelocities.npy')
velocities_newton_armijo = np.load('Newton_armijovelocities.npy')
velocities_picard_exact = np.load('Picard_exactvelocities.npy')
velocities_picard = np.load('Picardvelocities.npy')
coor = np.load('Picardcoordinates.npy')

top_grid_newton_exact = np.load('Newton_exacttop_grid.npy')
top_grid_newton_armijo = np.load('Newton_armijotop_grid.npy')
top_grid_picard_exact = np.load('Picard_exacttop_grid.npy')
top_grid_picard = np.load('Picardtop_grid.npy')

# We create a list for the indices of the relevant coordinates
# and a list of the relevant coordinates
(relevant_coord,x2) = relevant_coordinates(coor,top_grid_newton_exact)
# Next, we obtain the surface velocity calculates as in ISMIP-HOM
vel_norm_newton_exact = obtain_velocity(velocities_newton_exact,relevant_coord)
vel_norm_newton_armijo = obtain_velocity(velocities_newton_armijo,relevant_coord)
vel_norm_picard_exact = obtain_velocity(velocities_picard_exact,relevant_coord)
vel_norm_picard = obtain_velocity(velocities_picard,relevant_coord)


# We sorted the list of velocities by the x-coordinate
indices_sorted = np.argsort(x2)
coor_sorted = x2[indices_sorted]
coor_sorted = coor_sorted/5000.0

vel_norm_newton_exact_sorted = vel_norm_newton_exact[indices_sorted]
vel_norm_newton_armijo_sorted = vel_norm_newton_armijo[indices_sorted]
vel_norm_picard_exact_sorted = vel_norm_picard_exact[indices_sorted]
vel_norm_picard_sorted = vel_norm_picard[indices_sorted]


# Reerence solutions
vel_aas1 = np.abs(pd.read_csv('Reference_solutions/aas1e000_surf.txt',sep=" ",header=None))
vel_cma1 = np.abs(pd.read_csv('Reference_solutions/cma1e000.txt',sep=" ",header=None))
vel_mmr1 = np.abs(pd.read_csv('Reference_solutions/mmr1e000.txt',sep="\t",header=None))
vel_oga1 = np.abs(pd.read_csv('Reference_solutions/oga1e000.txt',sep=" ",header=None))
vel_spr1 = np.abs(pd.read_csv('Reference_solutions/spr1e000.txt',sep="\s+",header=None))
vel_ssu1 = np.abs(pd.read_csv('Reference_solutions/ssu1e000.txt',sep="\s+",header=None))
vel_yko1 = np.abs(pd.read_csv('Reference_solutions/yko1e000.txt',sep="\s+",header=None))

vel_yko1[0] = vel_yko1[0] / 5000.0

df_aas1_interpolated = np.interp(coor_sorted,vel_aas1[0],vel_aas1[1])
df_cma1_interpolated = np.interp(coor_sorted,vel_cma1[0],vel_cma1[1])
df_mmr1_interpolated = np.interp(coor_sorted,vel_mmr1[0],vel_mmr1[1])
df_oga1_interpolated = np.interp(coor_sorted,vel_oga1[0],vel_oga1[2])
df_spr1_interpolated = np.interp(coor_sorted,vel_spr1[0],vel_spr1[1])
df_ssu1_interpolated = np.interp(coor_sorted,vel_ssu1[0],vel_ssu1[1])
df_yko1_interpolated = np.interp(coor_sorted,vel_yko1[0],vel_yko1[1])

arr = np.array([df_aas1_interpolated,df_cma1_interpolated,df_mmr1_interpolated,df_oga1_interpolated,df_spr1_interpolated,df_ssu1_interpolated,df_yko1_interpolated])
std = np.std(arr,axis=0)
mean = np.mean(arr,axis=0)



font=14
width = 2

plt.rcParams['figure.figsize'] = (8,6)
# With *5000.0, we transform the plot back to the original length.
plt.plot(coor_sorted*5000.0,mean,color='grey',linewidth=width,label="Mean ISMIP-HOM simulations",alpha=0.6)
plt.plot(coor_sorted*5000.0,mean+std,'--',color='grey',linewidth=width,label="Mean+std ISMIP-HOM simulations",alpha=0.6)
plt.plot(coor_sorted*5000.0,mean-std,'--',color='grey',linewidth=width,label="Mean-std ISMIP-HOM simulations",alpha=0.6)


plt.plot(coor_sorted*5000.0,vel_norm_newton_exact_sorted,label='Newton exact',color='#1b9e77',linewidth=width,marker='x')
plt.plot(coor_sorted*5000.0,vel_norm_newton_armijo_sorted,label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.plot(coor_sorted*5000.0,vel_norm_picard_exact_sorted,label='Picard exact',color='#7570b3',linewidth=width,marker='x')
plt.plot(coor_sorted*5000.0,vel_norm_picard_sorted,label='Picard',color='black',linewidth=width)
plt.xlabel('Coordinate in x-direction in m',fontsize=font)
plt.ylabel('Velocity in m a$^{-1}$',fontsize=font)
plt.ylim(-10,80)
plt.legend()
plt.grid()
plt.show()


