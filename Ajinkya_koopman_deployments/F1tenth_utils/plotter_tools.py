import numpy as np
import matplotlib.pyplot as plt
from F1tenth_utils import lift_states
def plot_lifted_predictions(X,A,B,U,run_name,op_file_name=None):
    '''
    X is the list of trajectories
    A and B are the matrces obtained from EDMD
    U is the list of control sequences for the trajectories in X
    zt, zt_1 are the lifted X and Y matrices
    run_name is for title of the graphs indicating the type of data provided
    op_file_name is the name of the figure to be saved
    '''
    X0 = np.vstack(X)
    U = np.vstack(U)
    zt = lift_states.lift_states(X0)


    traj_points = [x.shape[0] for x in X]
    traj_points.insert(0,0)
    count = 0

    fig = plt.figure(constrained_layout=True, figsize=(12, len(traj_points)*2))
    subfigs = fig.subfigures(len(traj_points), 1)
    # fig, axs = plt.subplots(len(traj_points)-1, 3, tight_layout=True, figsize=(12, 20))

    for i in range(1,len(traj_points)):
        axs = subfigs[i-1].subplots(1, 3)
        # y_lift_test = zt_1[:,traj_points[i-1]:traj_points[i]]
        x_lift_test = zt[:,count:count+traj_points[i]]
        u_test = U[count:count+traj_points[i],:].T
        y_lifted = A@x_lift_test + B@u_test
        count += traj_points[i]
        # print('Koopman prediction and data comparison for run %d'%(i))

        subfigs[i-1].suptitle('Koopman prediction and data comparison for %s %d'%(run_name,i), fontsize=16)
        axs[0].plot(y_lifted[1,:], '--r')
        axs[0].plot(X[i-1][1:,0],'b', label='X actual')
        axs[0].set_title('Predicted X v/s True value')
        axs[0].set(ylabel=r'$x$',xlabel=r'$t$')
        axs[0].legend(['X_predicted','X_actual'])

        axs[1].plot(y_lifted[2,:], '--r', label='Y predicted')
        axs[1].plot(X[i-1][1:,1],'b',label='Y actual')
        axs[1].set_title('Predicted Y v/s True value')
        axs[1].set(ylabel=r'$y$',xlabel=r'$t$')
        axs[1].legend(['Y_predicted','Y_actual'])

        axs[2].plot(y_lifted[4,:], '--r', label='Theta predicted')
        axs[2].plot(X[i-1][1:,3],'b',label='Theta actual')
        axs[2].set_title('Predicted Theta v/s True value')
        axs[2].set(ylabel=r'$\phi$',xlabel=r'$t$')
        axs[2].legend(['Theta_predicted','Theta_actual'])

    if not op_file_name== None:
        file_name = op_file_name+'.jpg'
        plt.savefig(file_name)
        print('Figure_saved')
    else:
        print('Figure name not provided for save file')