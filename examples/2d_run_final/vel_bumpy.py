import matplotlib.pyplot as plt
import numpy as np


import pandas as pd


df_aas2 = pd.read_csv("ISMIP_b_005/aas2b005.txt",sep=" ",header=None)
df_cma1 = pd.read_csv("ISMIP_b_005/cma1b005.txt",sep=" ",header=None)
df_jvj1 = pd.read_csv("ISMIP_b_005/jvj1b005.txt",sep="\t",header=None)
df_mmr1 = pd.read_csv("ISMIP_b_005/mmr1b005.txt",sep="\t",header=None)
df_rhi3 = pd.read_csv("ISMIP_b_005/rhi3b005.txt",sep=" ",header=None)
df_ssu1 = pd.read_csv("ISMIP_b_005/ssu1b005.txt",sep="\t",header=None)




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
			#vel_field[k] =dataset[vel_index][0]
			vel_field[k] = float("NaN")
			
	return vel_field



velocities_Newton_exact = np.load('Newton_exactvelocities.npy')
velocities_Newton_Armijo = np.load('armijovelocities.npy')
velocities_Newton_Armijo_min_step = np.load('armijo_min_stepvelocities.npy')
velocities_Picard_exact = np.load('exact_picardvelocities.npy')
velocities_Picard_Constant = np.load('picardvelocities.npy')
velocities_Picard_Reference = np.load('reference_solutionvelocities.npy')
coor = np.load('picardcoordinates.npy')

# We create a list for the indices of the relevant coordinates
# and a list of the relevant coordinates
(relevant_coord,x2) = relevant_coordinates(coor)
#(relevant_coord_unrefined,x2_unrefined) = relevant_coordinates(coor_unrefined)
# Next, we obtain the surface velocity calculates as in ISMIP-HOM
vel_norm_Newton_exact = obtain_velocity(velocities_Newton_exact,relevant_coord)
vel_norm_Newton_Armijo = obtain_velocity(velocities_Newton_Armijo,relevant_coord)
vel_norm_Newton_Armijo_min_step = obtain_velocity(velocities_Newton_Armijo_min_step,relevant_coord)
vel_norm_Picard_exact = obtain_velocity(velocities_Picard_exact,relevant_coord)
vel_norm_Picard_Constant = obtain_velocity(velocities_Picard_Constant,relevant_coord)
vel_norm_Picard_Reference = obtain_velocity(velocities_Picard_Reference,relevant_coord)

plt.rcParams['figure.figsize'] = (8,6)


font=14
width = 2
plt.ylim(10.0,12.5)

num_interpolation_points = 80

df_cma1_100 = interpolate_to_new_grid(df_cma1,1,num_interpolation_points)
df_jvj1_100 = interpolate_to_new_grid(df_jvj1,1,num_interpolation_points)
df_mmr1_100 = interpolate_to_new_grid(df_mmr1,1,num_interpolation_points)
df_rhi3_100 = interpolate_to_new_grid(df_rhi3,2,num_interpolation_points)
df_ssu1_100 = interpolate_to_new_grid(df_ssu1,1,num_interpolation_points)
df_aas2_100 = interpolate_to_new_grid(df_aas2,1,num_interpolation_points)

arr = np.array([df_cma1_100,df_jvj1_100,df_mmr1_100,df_rhi3_100,df_ssu1_100,df_aas2_100])
std = np.std(arr,axis=0)
mean = np.mean(arr,axis=0)

x_values = np.linspace(0.0,1.0,num=num_interpolation_points+1)


plt.plot(x_values,mean,color='grey',linewidth=width,label="Mean ISMIP-HOM simulations",alpha=0.5)
plt.plot(x_values,mean+std,'--',color='grey',linewidth=width,label="Mean+std ISMIP-HOM simulations",alpha=0.5)
plt.plot(x_values,mean-std,'--',color='grey',linewidth=width,label="Mean-std ISMIP-HOM simulations",alpha=0.5)

plt.plot(x2/5000.0,vel_norm_Newton_Armijo,label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.plot(x2/5000.0,vel_norm_Newton_exact,label='Newton exact',color='#1b9e77',linewidth=width,marker='x')
plt.plot(x2/5000.0,vel_norm_Newton_Armijo_min_step,label='Newton Armijo min step',color='#1b9e77',linewidth=width,marker='+')
plt.plot(x2/5000.0,vel_norm_Picard_Constant,label='Picard',color='black',linewidth=width)
plt.plot(x2/5000.0,vel_norm_Picard_exact,label='Picard exact',color='#7570b3',linewidth=width,marker='x')



plt.rcParams["text.usetex"] = True
plt.xlabel('x position normalized',fontsize=font)
plt.ylabel('Velocity in m a$^{-1}$',fontsize=font)
plt.rcParams["text.usetex"] = False
plt.legend()
plt.grid()
plt.show()




plt.semilogy(x2/5000.0,(abs(vel_norm_Newton_exact-vel_norm_Picard_Reference))/vel_norm_Picard_Reference,label='Newton exact',color='#1b9e77',linewidth=width,marker='x')
plt.semilogy(x2/5000.0,(abs(vel_norm_Newton_Armijo-vel_norm_Picard_Reference))/vel_norm_Picard_Reference,label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(x2/5000.0,(abs(vel_norm_Newton_Armijo_min_step-vel_norm_Picard_Reference))/vel_norm_Picard_Reference,label='Newton Armijo min step',color='#1b9e77',linewidth=width,marker='+')
plt.semilogy(x2/5000.0,(abs(vel_norm_Picard_exact-vel_norm_Picard_Reference))/vel_norm_Picard_Reference,label='Picard exact',color='#7570b3',linewidth=width,marker='x',alpha=0.7)
plt.semilogy(x2/5000.0,(abs(vel_norm_Picard_Constant-vel_norm_Picard_Reference))/vel_norm_Picard_Reference,label='Picard',color='black',linewidth=width)

plt.xlabel('x position normalized',fontsize=font)
plt.ylabel('Relative difference',fontsize=font)
plt.legend(loc='center right', bbox_to_anchor=(1, 0.65)) 
plt.grid()
plt.show()

