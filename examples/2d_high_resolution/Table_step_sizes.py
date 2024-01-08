import numpy as np

Newton_Armijo_step_sizes = np.load("Newton_with_Armijostep_sizes.npy")
Newton_with_Armijo_min_stepstep_sizes = np.load("Newton_with_Armijo_min_stepstep_sizes.npy")
Newton_with_exactstep_sizes = np.load("Newton_with_exactstep_sizes.npy")
Picard_exactstep_sizes = np.load("Picard_exactstep_sizes.npy")


Newton_Armijo_step_sizes = np.insert(Newton_Armijo_step_sizes,0,0)
Newton_with_Armijo_min_stepstep_sizes = np.insert(Newton_with_Armijo_min_stepstep_sizes,0,0)
Newton_with_exactstep_sizes = np.insert(Newton_with_exactstep_sizes,0,0)
Picard_exactstep_sizes = np.insert(Picard_exactstep_sizes,0,0)

table = np.column_stack((np.arange(0,101),Newton_Armijo_step_sizes,Newton_with_Armijo_min_stepstep_sizes,Newton_with_exactstep_sizes,Picard_exactstep_sizes))


np.savetxt('step_sizes.csv',table,delimiter=',', fmt='%d, %e, %e, %e,%e')
