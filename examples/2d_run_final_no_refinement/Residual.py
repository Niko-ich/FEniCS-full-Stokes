import matplotlib.pyplot as plt
import numpy as np


font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)


newton_exact_norm = np.sqrt(np.load('Newton_with_exactnorm_list.npy'))
newton_armijo_norm = np.sqrt(np.load('Newton_with_Armijonorm_list.npy'))
newton_armijo_min_step_norm = np.sqrt(np.load('Newton_with_Armijo_min_stepnorm_list.npy'))
picard_exact_norm = np.sqrt(np.load('Picard_exactnorm_list.npy'))
picard_norm = np.sqrt(np.load('Picard_constantnorm_list.npy'))

plt.semilogy(newton_armijo_norm/picard_norm[0],label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_norm/picard_norm[0],label='Newton armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_norm/picard_norm[0],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_norm/picard_norm[0],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(picard_norm/picard_norm[0],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()



newton_exact_functional = np.load('Newton_with_exactfunctional_values_list.npy')
newton_armijo_functional = np.load('Newton_with_Armijofunctional_values_list.npy')
newton_armijo_min_step_functional = np.load('Newton_with_Armijo_min_stepfunctional_values_list.npy')
picard_exact_functional = np.load('Picard_exactfunctional_values_list.npy')
picard_constant_functional = np.load('Picard_constantfunctional_values_list.npy')

plt.plot(newton_armijo_functional[0:14],label='Newton armijo',color='#1b9e77',linewidth=width,marker='s')
plt.plot(newton_armijo_min_step_functional[0:14],label='Newton armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.plot(newton_exact_functional[0:14],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.plot(picard_exact_functional[0:14],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.plot(picard_constant_functional[0:14],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Functional value',fontsize=font)
plt.grid()
plt.show()
