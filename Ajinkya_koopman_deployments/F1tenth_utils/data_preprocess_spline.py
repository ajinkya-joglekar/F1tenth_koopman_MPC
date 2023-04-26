# from pysindy.differentiation import SmoothedFiniteDifference # Smoothen out the data
# from pysindy.differentiation import SpectralDerivative
# from pysindy.differentiation import FiniteDifference
import os
import numpy as np
import scipy.io as spio
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
from scipy.interpolate import PchipInterpolator

random.seed(0)

def preprocess(dir,plot_data=False):
    X = []
    X_orig = []
    Y = []
    U = []
    U_orig = []
    T_diff_states = []
    T_diff_inputs = []
    f_name = []


    for file in os.listdir(dir):
        # print('Processing file:',file)
        filename = os.path.join(dir, file)
        data = spio.loadmat(filename, squeeze_me=False)
        inputs = data['updated_data']['inputs'].item()
        inputs_ts = data['updated_data']['inputs_ts'].item()
        inputs_ts = np.reshape(inputs_ts, (np.shape(inputs_ts)[0],))
        states = data['updated_data']['states'].item()
        states_ts = data['updated_data']['states_ts'].item()
        states_ts = np.reshape(states_ts, np.shape(states_ts)[0])
        states[:, 2] = np.unwrap(states[:, 2])  # Unwrapping the optitrack measured angle

        # Use a common time difference for all observations
        t_diff_states = [states_ts[i + 1] - states_ts[0] for i in range(len(states_ts) - 1)]
        # t_diff_= [states_ts[i + 1] - states_ts[0] for i in range(len(states_ts) - 1)]
        t_diff_inputs = [inputs_ts[i + 1] - inputs_ts[0] for i in range(len(inputs_ts) - 1)]
        # t_diff.insert(0,0)

        t_diff_states = np.array(t_diff_states)
        t_diff_inputs = np.array(t_diff_inputs)

        if t_diff_states[-1] > 2:
            print('Processing file',file)

            #Spline based smoothing for states and control
            states_ts_range = round((t_diff_states[-1] - t_diff_states[0])*10)  # Number of samples to generate from the data
            inputs_ts_range = round((t_diff_inputs[-1] - t_diff_inputs[0])*10)
            # print(states_ts_range, inputs_ts_range)
            ts_states_spline = np.linspace(t_diff_states[0], t_diff_states[-1], states_ts_range)
            ts_inputs_spline = np.linspace(t_diff_inputs[0], t_diff_inputs[-1], inputs_ts_range)

            # Spline based sampling for all the states
            # Spline based sampling for X
            # print(t_diff_states.shape,states.shape)
            x_dot = np.diff(states[1:,0])/np.diff(t_diff_states)
            x_dot = np.insert(x_dot, 0, 0)

            # print(t_diff_states.shape,states[1:,0].shape,x_dot.shape)
            spl_x = PchipInterpolator(t_diff_states, states[1:,0].T)
            # spl_x.set_smoothing_factor(0) # Set the smoothing factor
            xs = spl_x(ts_states_spline) # Get the sampled x
            # print(xs.shape)
            # Spline based sampling for Y
            spl_y = PchipInterpolator(t_diff_states, states[1:, 1].T)
            # spl_y = UnivariateSpline(t_diff_states, states[1:,1].T)
            # spl_y.set_smoothing_factor(0) # Set the smoothing factor
            ys = spl_y(ts_states_spline) # Get the sampled y
            # print(ys.shape)
            # Spline based sampling for V
            dist = []
            for i in range(1, xs.shape[0]):
                dist_ = np.sqrt((xs[i] - xs[i-1]) ** 2 + (ys[i] - ys[i-1]) ** 2)
                dist.append(dist_)
            dist = np.array(dist)
            vel = dist/0.1
            vel = np.insert(vel, 0, 0)
            # print(vel.shape)

            # spl_v = UnivariateSpline(t_diff_states, vel[1:].T)
            # spl_v.set_smoothing_factor(0.5) # Set the smoothing factor
            # vs = spl_v(ts_states_spline) # Get the sampled vel

            # Spline based sampling for Theta
            spl_theta = PchipInterpolator(t_diff_states, states[1:,2].T)
            # spl_theta.set_smoothing_factor(0) # Set the smoothing factor
            theta_s = spl_theta(ts_states_spline) # Get the sampled theta

            # Spline based sampling for all control
            # Spline based sampling for v
            spl_v_ip = PchipInterpolator(t_diff_inputs, inputs[1:,0].T)
            # spl_v_ip.set_smoothing_factor(0.2) # Set the smoothing factor
            vp_ip_s = spl_v_ip(ts_states_spline) # Get the sampled vel input
            # vp_ip_s[vp_ip_s>5] = 0
            # vp_ip_s[vp_ip_s<5] = 0

            #Spline based sampling for delta
            spl_delta_ip = PchipInterpolator(t_diff_inputs, inputs[1:,1].T)
            # spl_delta_ip.set_smoothing_factor(0.01) # Set the smoothing factor
            delta_ip_s = spl_delta_ip(ts_states_spline) # Get the sampled delta input
            # delta_ip_s[delta_ip_s<-2] = 0
            # delta_ip_s[delta_ip_s>2] = 0

            x_train_ = np.column_stack((xs.T, ys.T, vel, theta_s.T))
            x_train_orig = np.column_stack((states[1:,0],states[1:,1],states[1:,2]))

            X.append(x_train_[:-1,:]) ## New change to take data from 0:n-1
            X_orig.append(x_train_orig)
            Y.append(x_train_[1:,:]) # Appending the 1:n data to Y

            u_train_ = np.column_stack((vp_ip_s.T, delta_ip_s.T))
            u_train_orig = np.column_stack((inputs[1:,0],inputs[1:,1]))
            # u_train_ = np.column_stack((inputs[:,0], inputs[:,1]))
            U.append(u_train_[:-1,:]) # Disregarding the last control input
            U_orig.append(u_train_orig)
            T_diff_states.append(t_diff_states)
            T_diff_inputs.append(t_diff_inputs)
            f_name.append(file)


    # order_ = np.arange(0, len(X))
    # # random.shuffle(order_)
    # X_shuffle = [X[i] for i in order_]
    # X_shuffle_orig = [X_orig[i] for i in order_]
    # Y_shuffle = [Y[i] for i in order_]
    # U_shuffle = [U[i] for i in order_]
    # U_shuffle_orig = [U_orig[i] for i in order_]
    # f_name_shuffle = [f_name[i] for i in order_]
    # # print(f_name_shuffle)
    # T_diff_states_shuffle = [T_diff_states[i] for i in order_]
    # T_diff_inputs_shuffle = [T_diff_inputs[i] for i in order_]


    return X, X_orig, Y, U, U_orig, T_diff_states,T_diff_inputs,f_name

