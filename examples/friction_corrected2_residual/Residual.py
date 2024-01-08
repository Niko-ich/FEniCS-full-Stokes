import matplotlib.pyplot as plt
import numpy as np


font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)


newton_exact_norm = np.sqrt(np.load('150x30Newton_with_exact_friction_coeff_high_res10000000.0norm_list.npy'))
newton_armijo_norm = np.sqrt(np.load('150x30Newton_with_Armijo_friction_coeff_high_res10000000.0norm_list.npy'))
newton_armijo_min_step_norm = np.sqrt(np.load('150x30Newton_with_Armijo_min_step_friction_coeff_high_res10000000.0norm_list.npy'))
picard_exact_norm = np.sqrt(np.load('150x30Picard_exact_friction_coeff_high_res10000000.0norm_list.npy'))
picard_norm = np.sqrt(np.load('150x30Picard_Reference_friction_coeff_high_res10000000.0norm_list.npy'))

plt.semilogy(newton_armijo_norm[0:33]/picard_norm[0],label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_norm[0:33]/picard_norm[0],label='Newton armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_norm[0:33]/picard_norm[0],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_norm[0:33]/picard_norm[0],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.6)
plt.semilogy(picard_norm[0:33]/picard_norm[0],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.6)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()


newton_exact_norm = np.sqrt(np.load('150x30Newton_with_exact_friction_coeff_high_res1000.0norm_list.npy'))
newton_armijo_norm = np.sqrt(np.load('150x30Newton_with_Armijo_friction_coeff_high_res1000.0norm_list.npy'))
newton_armijo_min_step_norm = np.sqrt(np.load('150x30Newton_with_Armijo_min_step_friction_coeff_high_res1000.0norm_list.npy'))
picard_exact_norm = np.sqrt(np.load('150x30Picard_exact_friction_coeff_high_res1000.0norm_list.npy'))
picard_norm = np.sqrt(np.load('150x30Picard_Reference_friction_coeff_high_res1000.0norm_list.npy'))

plt.semilogy(newton_armijo_norm[0:33]/picard_norm[0],label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_norm[0:33]/picard_norm[0],label='Newton armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_norm[0:33]/picard_norm[0],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_norm[0:33]/picard_norm[0],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.6)
plt.semilogy(picard_norm[0:33]/picard_norm[0],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.6)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()
