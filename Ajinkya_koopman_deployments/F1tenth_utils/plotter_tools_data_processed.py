import matplotlib.pyplot as plt


def plot_processed_data(X,X_orig,U,U_orig,T_diff_states,T_diff_ip,f_name):
# Plot the data
    fig = plt.figure(constrained_layout=True, figsize=(12, len(X)*2))
    subfigs = fig.subfigures(len(X), 1)
# fig, axs = plt.subplots(5, 1, tight_layout=True, figsize=(12, 8))
    for i in range(len(X)):
        # T = T_diff_states[i]*10
        # T_ip = T_diff_ip[i]*10
        axs = subfigs[i-1].subplots(1, 5)
        subfigs[i-1].suptitle('Smoothened input comparison for %s'%(f_name[i]), fontsize=16)
        x = X[i][:,0]
        x_orig = X_orig[i][:,0]
        y = X[i][:,1]
        y_orig = X_orig[i][:,1]
        # if min(y) < -10:
        #     print(f_name[i])
        phi = X[i][:,3]
        phi_orig = X_orig[i][:,2]
        vel = U[i][:,0]
        vel_orig = U_orig[i][:,0]
        delta = U[i][:,1]
        delta_orig = U_orig[i][:,1]

        # axs[0].plot(T, x_orig, 'o', ms=2)
        axs[0].plot(x, '--c')
        axs[0].set(ylabel=r'$x$',xlabel=r't')
        axs[0].legend(['X_orig','X_smooth'])

        # axs[1].plot(T, y_orig, 'o', ms=2)
        axs[1].plot(y, '--c')
        axs[1].set(ylabel=r'$y$',xlabel=r't')
        axs[1].legend(['Y_orig','Y_smooth'])

        # axs[2].plot(T, phi_orig, 'o', ms=2)
        axs[2].plot(phi, '--c')
        axs[2].set(ylabel=r'$phi$',xlabel=r't')
        axs[2].legend(['phi_orig','phi_smooth'])

        # axs[3].plot(T_ip, vel_orig, 'o', ms=2)
        axs[3].plot(vel, '--c')
        axs[3].set(ylabel=r'$vel$',xlabel=r't')
        axs[3].legend(['vel_orig','vel_smooth'])

        # axs[4].plot(T_ip, delta_orig, 'o', ms=2)
        axs[4].plot(delta,'--c')
        axs[4].set(ylabel=r'$\delta$',xlabel=r'$t$')
        axs[4].legend(['delta_orig','delta_smooth'])