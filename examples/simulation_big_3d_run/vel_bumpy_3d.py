import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def relevant_coordinates(coor):
	relevant_coord = list()
	for k in range(0,np.shape(coor)[0]):
		if(abs(coor[k][2]-1000.0)<1.0 and abs(coor[k][1]-1250.0)<0.5 and coor[k][0]>-1.0 and coor[k][0]<5001.0):
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
		vel_norm[counter] = np.sqrt((vel[k][0]*np.cos(alpha))**2-(vel[k][2]*np.sin(alpha))**2+vel[k][1]**2)
		counter = counter+1
	return vel_norm


def obtain_velocity_datasets(dataset):
    # We assume that data is available at y=L/4.
    indices = np.where(dataset[1][:]==0.25)
    dataset_list = np.zeros((len(indices[0]),2))
    for k in range(0,len(indices[0])):
        dataset_list[k,0] = dataset[0][indices[0][k]]
        dataset_list[k,1] = np.sqrt(dataset[2][indices[0][k]]*dataset[2][indices[0][k]]+dataset[3][indices[0][k]]*dataset[3][indices[0][k]])

    # For the aas2a dataset
    if(len(indices[0])==0):
        indices1 = np.where(dataset[1][:] == 0.2375)
        indices2 = np.where(dataset[1][:] == 0.2625)
        if(len(indices1[0])==0):
            indices1 = np.where(dataset[1][:] == 0.231707)
            indices2 = np.where(dataset[1][:] == 0.256098)
        dataset_list = np.zeros((len(indices1[0]),2))
        for k in range(0,len(indices1[0])):
            if(dataset[0][indices1[0][k]]==dataset[0][indices2[0][k]]):
                dataset_list[k,0] = dataset[0][indices1[0][k]]
                total_weight = abs(dataset[1][indices1[0][k]]-0.25) + abs(dataset[1][indices2[0][k]]-0.25)
                weight_a = abs(dataset[1][indices1[0][k]]-0.25)/total_weight
                weight_b = abs(dataset[1][indices2[0][k]]-0.25)/total_weight
                vx = dataset[2][indices1[0][k]]*weight_a+dataset[2][indices2[0][k]]*weight_b
                vy = dataset[3][indices1[0][k]]*weight_a+dataset[3][indices2[0][k]]*weight_b
                dataset_list[k,1] = np.sqrt(vx*vx+vy*vy)
            else:
                print('Problem')

    return dataset_list


def interpolate_to_new_grid(dataset,vel_index,new_size):
	# velocity_index: the index with the velocity is dependent on the dataset.
	# new_size: size of the new field.
	vel_field = np.zeros(new_size+1)
	for k in range(0,new_size+1):
		index = 0
		while(dataset[index][0]<= k*1.0/new_size and index<np.shape(dataset)[0]-1):
			index = index + 1
		if(index>0 and k*1.0/new_size<dataset[index][0]):
			index = index - 1
			# Now we have dataset[0][index] <= k*1.0/new_size < dataset[0][index+1]
			weight_left = 1-abs(dataset[index][0]-k*1.0/new_size)/(dataset[index+1][0]-dataset[index][0])
			weight_right = 1-abs(dataset[index+1][0]-k*1.0/new_size)/(dataset[index+1][0]-dataset[index][0])
			vel_field[k] = weight_left*dataset[index][vel_index]+weight_right*dataset[index+1][vel_index]
		elif(dataset[index][0]==1):
			vel_field[k] = dataset[index][vel_index]
		elif(index ==np.shape(dataset)[0]-1):
			vel_field[k] = float("NaN")
		else:
			vel_field[k] = float("NaN")

	return vel_field



df_aas2 = pd.read_csv("ISMIP_a/aas2a005.txt",sep=" ",header=None)
df_cma1 = pd.read_csv("ISMIP_a/cma1a005.txt",sep=" ",header=None)
df_oga1 = pd.read_csv("ISMIP_a/oga1a005.txt",sep="  ",header=None)
df_rhi1 = pd.read_csv("ISMIP_a/rhi1a005.txt",sep=" ",header=None)
df_rhi3 = pd.read_csv("ISMIP_a/rhi3a005.txt",sep=" ",header=None)


dataset_cma = obtain_velocity_datasets(df_cma1)
dataset_oga = obtain_velocity_datasets(df_oga1)
dataset_aas = obtain_velocity_datasets(df_aas2)
dataset_rhi1 = obtain_velocity_datasets(df_rhi1)
dataset_rhi3 = obtain_velocity_datasets(df_rhi3)

num_interpolation_points = 80

dataset_cma_new_grid = interpolate_to_new_grid(dataset_cma,1,num_interpolation_points)
dataset_oga_new_grid = interpolate_to_new_grid(dataset_oga,1,num_interpolation_points)
dataset_aas_new_grid = interpolate_to_new_grid(dataset_aas,1,num_interpolation_points)
dataset_rhi1_new_grid = interpolate_to_new_grid(dataset_rhi1,1,num_interpolation_points)
dataset_rhi3_new_grid = interpolate_to_new_grid(dataset_rhi3,1,num_interpolation_points)

arr = np.array([dataset_cma_new_grid,dataset_oga_new_grid,dataset_aas_new_grid,dataset_rhi1_new_grid,dataset_rhi3_new_grid])
mean = np.mean(arr,axis=0)
std = np.std(arr,axis=0)

x_values = np.linspace(0.0,1.0,num=num_interpolation_points+1)

# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)

font=14
width = 2

plt.plot(x_values,mean,color='grey',linewidth=width,label="Mean ISMIP-HOM simulations",alpha=0.5)
plt.plot(x_values,mean+std,'--',color='grey',linewidth=width,label="Mean+std ISMIP-HOM simulations",alpha=0.5)
plt.plot(x_values,mean-std,'--',color='grey',linewidth=width,label="Mean-std ISMIP-HOM simulations",alpha=0.5)


vel_ref_picard = np.load('ref_solutionvelocities.npy')
vel_picard = np.load('Picard_constantvelocities.npy')
vel_picard_exact = np.load('exact_picardvelocities.npy')
vel_newton_exact = np.load('Newton_exactvelocities.npy')
vel_newton_armijo = np.load('Newton_with_armijovelocities.npy')
vel_newton_armijo_min_step = np.load('Newton_armijo_min_stepvelocities.npy')
coor = np.load('ref_solutioncoordinates.npy')

(relevant_coord,x2) = relevant_coordinates(coor)

vel_ref_picard = obtain_velocity(vel_ref_picard,relevant_coord)
vel_picard = obtain_velocity(vel_picard,relevant_coord)
vel_picard_exact = obtain_velocity(vel_picard_exact,relevant_coord)
vel_newton_exact = obtain_velocity(vel_newton_exact,relevant_coord)
vel_newton_armijo = obtain_velocity(vel_newton_armijo,relevant_coord)
vel_newton_armijo_min_step = obtain_velocity(vel_newton_armijo_min_step,relevant_coord)

plt.ylim((13.75,14.75))
plt.plot(x2/5000.0,vel_newton_armijo,label='Newton Armijo step sizes',color='#1b9e77',linewidth=width,marker='s')
plt.plot(x2/5000.0,vel_newton_exact,label='Newton exact step sizes',color='#1b9e77',linewidth=width,marker='x')
plt.plot(x2/5000.0,vel_newton_armijo_min_step,label='Newton Armijo min_step=0.5 sizes',color='#1b9e77',linewidth=width,marker='+')
plt.plot(x2/5000.0,vel_picard,label='Picard',color='black',linewidth=width)
plt.plot(x2/5000.0,vel_picard_exact,label='Picard exact step sizes',color='#7570b3',linewidth=width,marker='x')
plt.grid()


plt.xlabel('x position normalized',fontsize=font)
plt.rcParams["text.usetex"] = True
plt.ylabel('Velocity in m a$^{-1}$',fontsize=font)
plt.rcParams["text.usetex"] = False
plt.legend()
plt.show()

plt.semilogy(x2/5000.0,(abs(vel_newton_exact-vel_ref_picard))/vel_ref_picard,label='Newton exact',color='#1b9e77',linewidth=width,marker='x')
plt.semilogy(x2/5000.0,(abs(vel_newton_armijo-vel_ref_picard))/vel_ref_picard,label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(x2/5000.0,(abs(vel_newton_armijo_min_step-vel_ref_picard))/vel_ref_picard,label='Newton Armijo min step',color='#1b9e77',linewidth=width,marker='+')
plt.semilogy(x2/5000.0,(abs(vel_picard_exact-vel_ref_picard))/vel_ref_picard,label='Picard exact',color='#7570b3',linewidth=width,marker='x',alpha=0.7)
plt.semilogy(x2/5000.0,(abs(vel_picard-vel_ref_picard))/vel_ref_picard,label='Picard',color='black',linewidth=width)

plt.xlabel('x position normalized',fontsize=font)
plt.ylabel('Relative difference',fontsize=font)
plt.legend(loc='center left',bbox_to_anchor=(0, 0.23)) 
plt.grid()
plt.show()
