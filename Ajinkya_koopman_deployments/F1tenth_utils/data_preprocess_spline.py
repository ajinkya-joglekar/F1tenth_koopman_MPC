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

random.seed(0)

def preprocess(dir):
    X = []
    Y = []
    U = []
    T_diff_states = []
    T_diff_inputs = []


    for file in os.listdir(dir):
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
        t_diff_= [states_ts[i + 1] - states_ts[0] for i in range(len(states_ts) - 1)]
        t_diff_inputs = [inputs_ts[i + 1] - inputs_ts[0] for i in range(len(inputs_ts) - 1)]
        # t_diff.insert(0,0)
        t_diff_states = np.array(t_diff_states)
        t_diff_inputs = np.array(t_diff_inputs)

        if t_diff_states.shape[0] > 40:
            dist = [np.sqrt((states[i, 0] - states[0, 0]) ** 2 + (states[i, 1] - states[0, 1]) ** 2) for i in range(1, states.shape[0])]
            dist = np.array(dist)
            vel = dist / t_diff_
            vel = np.insert(vel, 0, 0)

            #Spline based smoothing for states and control
            states_ts_range = int(t_diff_states[-1] - t_diff_states[0])  # Number of samples to generate from the data
            ts_states_spline = np.linspace(0, t_diff_states[-1], states_ts_range * 10)

            # Spline based sampling for all the states
            # Spline based sampling for X
            # print(t_diff_states.shape,states.shape)
            spl_x = UnivariateSpline(t_diff_states, states[1:,0].T)
            spl_x.set_smoothing_factor(0.5) # Set the smoothing factor
            xs = spl_x(ts_states_spline) # Get the sampled x
            # Spline based sampling for Y
            spl_y = UnivariateSpline(t_diff_states, states[1:,1].T)
            spl_y.set_smoothing_factor(0.5) # Set the smoothing factor
            ys = spl_y(ts_states_spline) # Get the sampled y
            # Spline based sampling for V
            spl_v = UnivariateSpline(t_diff_states, vel[1:].T)
            spl_v.set_smoothing_factor(0.5) # Set the smoothing factor
            vs = spl_v(ts_states_spline) # Get the sampled vel
            # Spline based sampling for Theta
            spl_theta = UnivariateSpline(t_diff_states, states[1:,2].T)
            spl_theta.set_smoothing_factor(0.5) # Set the smoothing factor
            theta_s = spl_theta(ts_states_spline) # Get the sampled theta

            # Spline based sampling for all control
            # Spline based sampling for v
            spl_v_ip = UnivariateSpline(t_diff_states, inputs[1:,0].T)
            spl_v_ip.set_smoothing_factor(0.2) # Set the smoothing factor
            vp_ip_s = spl_v_ip(ts_states_spline) # Get the sampled vel input

            #Spline based sampling for delta
            spl_delta_ip = UnivariateSpline(t_diff_states, inputs[1:,1].T)
            spl_delta_ip.set_smoothing_factor(0.2) # Set the smoothing factor
            delta_ip_s = spl_delta_ip(ts_states_spline) # Get the sampled delta input

            x_train_ = np.column_stack((xs.T, ys.T, vs.T, theta_s.T))
            X.append(x_train_[0:-1,:]) ## New change to take data from 0:n-1
            Y.append(x_train_[1:,:]) # Appending the 1:n data to Y
            u_train_ = np.column_stack((vp_ip_s.T, delta_ip_s.T))
            U.append(u_train_[:-1,:]) # Disregarding the last control input
            T_diff_states.append(t_diff_states)
            T_diff_inputs.append(t_diff_inputs)

    order_ = np.arange(0, len(X))
    random.shuffle(order_)
    X_shuffle = [X[i] for i in order_]
    Y_shuffle = [Y[i] for i in order_]
    U_shuffle = [U[i] for i in order_]
    T_diff_states_shuffle = [T_diff_states[i] for i in order_]
    T_diff_inputs_shuffle = [T_diff_inputs[i] for i in order_]


    return X_shuffle, Y_shuffle, U_shuffle, T_diff_states_shuffle,T_diff_inputs_shuffle

