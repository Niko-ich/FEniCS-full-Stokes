import matplotlib.pyplot as plt
import numpy as np


font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)

newton_armijo_70x5 = np.sqrt(np.load('Newton_with_Armijo_70x5norm_list.npy'))
newton_armijo_140x10 = np.sqrt(np.load('Newton_with_Armijo_140x10norm_list.npy'))
newton_armijo_210x15 = np.sqrt(np.load('Newton_with_Armijo_210x15norm_list.npy'))
newton_armijo_280x20 = np.sqrt(np.load('Newton_with_Armijo_280x20norm_list.npy'))
newton_armijo_420x30 = np.sqrt(np.load('Newton_with_Armijo_420x30norm_list.npy'))
newton_armijo_560x40 = np.sqrt(np.load('Newton_with_Armijo_560x40norm_list.npy'))

plt.semilogy(newton_armijo_70x5[0:15]/newton_armijo_70x5[0],label='Newton armijo 70x5',color='#1b9e77',linewidth=width,marker='s')
plt.semilogy(newton_armijo_140x10[0:15]/newton_armijo_140x10[0],label='Newton armijo 140x10',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_armijo_210x15[0:15]/newton_armijo_210x15[0],label='Newton armijo 210x15',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(newton_armijo_280x20[0:15]/newton_armijo_280x20[0],label='Newton armijo 280x20',color='#7570b3',linewidth=width,linestyle='dotted',marker='x',alpha=0.7)
plt.semilogy(newton_armijo_420x30[0:15]/newton_armijo_420x30[0], label='Newton armijo 420x30',color='#7570b3',linestyle='dashed',linewidth=width,marker='*',alpha=0.7)
plt.semilogy(newton_armijo_560x40[0:15]/newton_armijo_560x40[0], label='Newton armijo 560x40',color='#7570b3',linewidth=width,marker='s',alpha=0.7)
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()

