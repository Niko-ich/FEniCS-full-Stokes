import numpy as np
import matplotlib.pyplot as plt

Newton_Armijo_time_iteration = np.load("Newton_armijocomputation_time_iteration.npy")
Newton_Armijo_min_step_time_iteration = np.load("Newton_armijo_min_stepcomputation_time_iteration.npy")
Newton_exact_time_iteration = np.load("Newton_exactcomputation_time_iteration.npy")
Picard_time_iteration = np.load("Picard_constantcomputation_time_iteration.npy")
Picard_exact_time_iteration = np.load("Picard_exactcomputation_time_iteration.npy")


Newton_Armijo_time_step_size = np.load("Newton_armijocomputation_time_step_size.npy")
Newton_Armijo_min_step_time_step_size = np.load("Newton_armijo_min_stepcomputation_time_step_size.npy")
Picard_exact_time_step_size = np.load("Picard_exactcomputation_time_step_size.npy")
Newton_exact_time_step_size = np.load("Newton_exactcomputation_time_step_size.npy")

# We calculate mean time and std for the iterations
print('Newton Armijo; mean:',np.mean(Newton_Armijo_time_iteration),'std:',np.std(Newton_Armijo_time_iteration))
print('Newton Armijo min step; mean:',np.mean(Newton_Armijo_min_step_time_iteration),'std:',np.std(Newton_Armijo_min_step_time_iteration))
print('Newton exact; mean:', np.mean(Newton_exact_time_iteration),'std:',np.std(Newton_exact_time_iteration))
print('Picard; mean:',np.mean(Picard_time_iteration),'std:',np.std(Picard_time_iteration))
print('Picard exact; mean:',np.mean(Picard_exact_time_iteration),'std:',np.std(Picard_exact_time_iteration))

print('Step size times')
# We calculate mean time and std for the step size controls
print('Newton Armijo; mean:',np.mean(Newton_Armijo_time_step_size),'std:',np.std(Newton_Armijo_time_step_size))
print('Newton Armijo min step; mean:',np.mean(Newton_Armijo_min_step_time_step_size),'std:',np.std(Newton_Armijo_min_step_time_step_size))
print('Newton exact; mean:', np.mean(Newton_exact_time_step_size),'std:',np.std(Newton_exact_time_step_size))
print('Picard exact; mean:',np.mean(Picard_exact_time_step_size),'std:',np.std(Picard_exact_time_step_size))


