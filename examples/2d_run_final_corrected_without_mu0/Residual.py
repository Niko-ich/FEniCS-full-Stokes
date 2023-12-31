import matplotlib.pyplot as plt
import numpy as np


# We plot the relative error
newton_exact_rel_error = np.sqrt(np.load('Newton_with_exact_without_mu0relative_error.npy'))
newton_armijo_rel_error = np.sqrt(np.load('Newton_with_Armijo_without_mu0relative_error.npy'))
newton_armijo_min_step_rel_error = np.sqrt(np.load('Newton_with_Armijo_min_step_without_mu0relative_error.npy'))
picard_exact_rel_error = np.sqrt(np.load('Picard_exact_without_mu0relative_error.npy'))
picard_armijo_rel_error = np.sqrt(np.load('Picard_armijo_without_mu0relative_error.npy'))
picard_rel_error = np.sqrt(np.load('Picard_constant_without_mu0relative_error.npy'))

font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)

plt.semilogy(picard_armijo_rel_error,label='Picard Armijo',color='#7570b3',linewidth=width,marker='s')
plt.semilogy(newton_armijo_rel_error,label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_rel_error,label='Newton Armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_rel_error,label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_rel_error,label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(picard_rel_error,label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative difference',fontsize=font)
plt.grid()
plt.show()

plt.semilogy(newton_armijo_rel_error[0:10],label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_rel_error[0:10],label='Newton Armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_rel_error[0:10],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_rel_error[0:10],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(picard_rel_error[0:10],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative difference',fontsize=font)
plt.grid()
plt.show()



newton_exact_rel_local_error = np.sqrt(np.load('Newton_with_exact_without_mu0relative_local_error.npy'))
newton_armijo_rel_local_error = np.sqrt(np.load('Newton_with_Armijo_without_mu0relative_local_error.npy'))
newton_armijo_min_step_rel_local_error = np.sqrt(np.load('Newton_with_Armijo_min_step_without_mu0relative_local_error.npy'))
picard_exact_rel_local_error = np.sqrt(np.load('Picard_exact_without_mu0relative_local_error.npy'))
picard_armijo_rel_local_error = np.sqrt(np.load('Picard_armijo_without_mu0relative_local_error.npy'))
picard_rel_local_error = np.sqrt(np.load('Picard_constant_without_mu0relative_local_error.npy'))

plt.semilogy(picard_armijo_rel_local_error,label='Picard Armijo',color='#7570b3',linewidth=width,marker='s')
plt.semilogy(newton_armijo_rel_local_error,label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_rel_local_error,label='Newton Armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_rel_local_error,label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_rel_local_error,label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(picard_rel_local_error,label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative local difference',fontsize=font)
plt.grid()
plt.show()

plt.semilogy(picard_armijo_rel_local_error[0:10],label='Picard Armijo',color='#7570b3',linewidth=width,marker='s')
plt.semilogy(newton_armijo_rel_local_error[0:10],label='Newton Armijo',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_min_step_rel_local_error[0:10],label='Newton Armijo min step 0.5',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_exact_rel_local_error[0:10],label='Newton exact',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(picard_exact_rel_local_error[0:10],label='Picard exact',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(picard_rel_local_error[0:10],label='Picard',color='#7570b3',linewidth=width,marker='*',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative local difference',fontsize=font)
plt.grid()
plt.show()



