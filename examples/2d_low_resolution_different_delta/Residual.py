import matplotlib.pyplot as plt
import numpy as np


font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)


newton_armijo_delta1e_minus_4_norm = np.sqrt(np.load('Newton_with_Armijo_delta0.0001norm_list.npy'))
newton_armijo_min_step_delta1e_minus_4_norm = np.sqrt(np.load('Newton_with_Armijo_min_step_delta0.0001norm_list.npy'))
newton_exact_delta1e_minus_4_norm = np.sqrt(np.load('Newton_with_exact_delta0.0001norm_list.npy'))
picard_constant_delta1e_minus_4_norm = np.sqrt(np.load('Picard_constant_delta0.0001norm_list.npy'))
picard_exact_delta1e_minus_4_norm = np.sqrt(np.load('Picard_exact_delta0.0001norm_list.npy'))


plt.semilogy(newton_armijo_delta1e_minus_4_norm[0:25]/picard_constant_delta1e_minus_4_norm[0],label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_delta1e_minus_4_norm[0:25]/picard_constant_delta1e_minus_4_norm[0],label='Newton armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_delta1e_minus_4_norm[0:25]/picard_constant_delta1e_minus_4_norm[0],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_constant_delta1e_minus_4_norm[0:25]/picard_constant_delta1e_minus_4_norm[0],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.6)
plt.semilogy(picard_exact_delta1e_minus_4_norm[0:25]/picard_constant_delta1e_minus_4_norm[0],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.6)

plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()


newton_armijo_delta1e_minus_3_norm = np.sqrt(np.load('Newton_with_Armijo_delta0.001norm_list.npy'))
newton_armijo_delta1e_minus_4_norm = np.sqrt(np.load('Newton_with_Armijo_delta0.0001norm_list.npy'))
newton_armijo_delta1e_minus_5_norm = np.sqrt(np.load('Newton_with_Armijo_delta1e-12norm_list.npy'))

plt.semilogy(newton_armijo_delta1e_minus_3_norm[0:25]/newton_armijo_delta1e_minus_3_norm[0],label='Newton armijo $\delta=10^{-3}$',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(newton_armijo_delta1e_minus_4_norm[0:25]/newton_armijo_delta1e_minus_4_norm[0],label='Newton armijo $\delta=10^{-4}$',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_delta1e_minus_5_norm[0:25]/newton_armijo_delta1e_minus_5_norm[0],label='Newton armijo $\delta=10^{-12}$',color='#1b9e77',linestyle='dashed',marker='+')
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()


