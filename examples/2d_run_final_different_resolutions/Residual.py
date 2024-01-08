import matplotlib.pyplot as plt
import numpy as np


# We plot the relative error
newton_armijo_rel_error_resx24 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx24relative_error.npy'))
newton_armijo_rel_error_resx48 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx48relative_error.npy'))
newton_armijo_rel_error_resx96 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx96relative_error.npy'))
newton_armijo_rel_error_resx192 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx192relative_error.npy'))
newton_armijo_rel_error_resx384 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx384relative_error.npy'))


#print('hsize',np.load('hmax_with_resx25.npy'))
hsize = []
for k in range(0,5):
    hsize.append(np.load('hmax_with_resx'+str(24*2**k)+'.npy'))

print('sizes',hsize)

last_resx24 = newton_armijo_rel_error_resx24[-1]
last_resx48 = newton_armijo_rel_error_resx48[-1]
last_resx96 = newton_armijo_rel_error_resx96[-1]
last_resx192 = newton_armijo_rel_error_resx192[-1]
last_resx384 = newton_armijo_rel_error_resx384[-1]

conv_rate = np.zeros(4)
conv_rate[0] = last_resx24/last_resx48 * hsize[1]/hsize[0]
conv_rate[1] = last_resx48/last_resx96 * hsize[2]/hsize[1]
conv_rate[2] = last_resx96/last_resx192 * hsize[3]/hsize[2]
conv_rate[3] = last_resx192/last_resx384 * hsize[4]/hsize[3]

print('conv_rates',conv_rate)

font=14
width = 2
# We set the size of the figures.
plt.rcParams['figure.figsize'] = (8,6)


plt.semilogy(newton_armijo_rel_error_resx24,label='25 grid points in x-direction',color='#1b9e77',linewidth=width)
plt.semilogy(newton_armijo_rel_error_resx48,label='49 grid points in x-direction',color='#1b9e77',linewidth=width,marker='x')
plt.semilogy(newton_armijo_rel_error_resx96,label='97 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(newton_armijo_rel_error_resx192,label='193 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_armijo_rel_error_resx384,label='385 grid points in x-direction',color='#7570b3',linewidth=width,alpha=0.6,linestyle='dashed',marker='x')
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative difference',fontsize=font)
plt.grid()
plt.show()




newton_armijo_rel_local_error_resx24 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx24relative_local_error.npy'))
newton_armijo_rel_local_error_resx48 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx48relative_local_error.npy'))
newton_armijo_rel_local_error_resx96 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx96relative_local_error.npy'))
newton_armijo_rel_local_error_resx192 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx192relative_local_error.npy'))
newton_armijo_rel_local_error_resx384 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx384relative_local_error.npy'))

plt.semilogy(newton_armijo_rel_local_error_resx24,label='25 grid points in x-direction',color='#1b9e77',linewidth=width)
plt.semilogy(newton_armijo_rel_local_error_resx48,label='49 grid points in x-direction',color='#1b9e77',linewidth=width,marker='x')
plt.semilogy(newton_armijo_rel_local_error_resx96,label='97 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(newton_armijo_rel_local_error_resx192,label='193 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_armijo_rel_local_error_resx384,label='385 grid points in x-direction',color='#7570b3',linewidth=width,alpha=0.6,linestyle='dashed',marker='x')
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative local difference',fontsize=font)
plt.grid()
plt.show()




newton_armijo_norm_resx24 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx24norm_list.npy'))
newton_armijo_norm_resx48 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx48norm_list.npy'))
newton_armijo_norm_resx96 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx96norm_list.npy'))
newton_armijo_norm_resx192 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx192norm_list.npy'))
newton_armijo_norm_resx384 = np.sqrt(np.load('Newton_with_Armijo_without_mu0_resx384norm_list.npy'))

plt.semilogy(newton_armijo_norm_resx24/newton_armijo_norm_resx24[0],label='24 grid points in x-direction',color='#1b9e77',linewidth=width)
plt.semilogy(newton_armijo_norm_resx48/newton_armijo_norm_resx48[0],label='49 grid points in x-direction',color='#1b9e77',linewidth=width,marker='x')
plt.semilogy(newton_armijo_norm_resx96/newton_armijo_norm_resx96[0],label='96 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dotted',marker='x')
plt.semilogy(newton_armijo_norm_resx192/newton_armijo_norm_resx192[0],label='192 grid points in x-direction',color='#1b9e77',linewidth=width,linestyle='dashed',marker='+')
plt.semilogy(newton_armijo_norm_resx384/newton_armijo_norm_resx384[0],label='384 grid points in x-direction',color='#7570b3',linewidth=width,alpha=0.6,linestyle='dashed',marker='x')
plt.legend()
plt.xlabel('Iteration number',fontsize=font)
plt.ylabel('Relative residual norm',fontsize=font)
plt.grid()
plt.show()
