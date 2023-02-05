# from pysindy.differentiation import SmoothedFiniteDifference # Smoothen out the data
# from pysindy.differentiation import SpectralDerivative
# from pysindy.differentiation import FiniteDifference
import os
import numpy as np
import scipy.io as spio
import random

random.seed(0)

def preprocess(dir):
    X = []
    U = []
    T_diff = []
    # sfd = SmoothedFiniteDifference(smoother_kws={'window_length': 10})
    # fd = FiniteDifference()
    # sd = SpectralDerivative()


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
        t_diff = [states_ts[i + 1] - states_ts[0] for i in range(len(states_ts) - 1)]
        # t_diff.insert(0,0)
        t_diff = np.array(t_diff)
        if t_diff.shape[0] > 45:
            dist = [np.sqrt((states[i, 0] - states[0, 0]) ** 2 + (states[i, 1] - states[0, 1]) ** 2) for i in
                    range(1, states.shape[0])]
            dist = np.array(dist)
            vel = dist / t_diff
            vel = np.insert(vel, 0, 0)
            x_train_ = np.column_stack((states[:, 0], states[:, 1], vel, states[:, 2]))
            X.append(x_train_)
            u_train_ = np.column_stack((inputs[:, 0], inputs[:, 1]))
            U.append(u_train_)
            T_diff.append(t_diff)

    order_ = np.arange(0, len(X))
    random.shuffle(order_)
    X_shuffle = [X[i] for i in order_]
    U_shuffle = [U[i] for i in order_]
    T_diff_shuffle = [T_diff[i] for i in order_]


    return X_shuffle, U_shuffle, T_diff_shuffle

