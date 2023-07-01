# importing the required module

import matplotlib.pyplot as plt
import numpy as np
import math
from least_square import *

def weighted_LS(X, Y, Q):
    s = X.transpose()
    a = np.linalg.inv(s.dot(Q).dot(X))
    b = s.dot(Q).dot(Y)
    theta = np.dot(a, b)
    Y_pred = np.dot(X, theta)
    error = Y - Y_pred
    NRMSE = math.sqrt((1 / N) * (np.dot(error.transpose(), error)))
    return theta, NRMSE

N = 20
n = 2
U = np.linspace(0, 1, N)
y = (4*(U-0.5)**2)
noise = np.random.normal(0, .1, y.shape) * 3*(U+0.01)
Y = y + noise
X = Regressor(U, n)
Q = np.identity(N) / (3*(U+0.01))
thetaW, NRMSEW = weighted_LS(X, Y, Q)
print("For weighted least square:", "\n", "theta = ", thetaW, "\n", "NRMSE = ",  NRMSEW)
theta, NRMSE = least_sqr(X, Y)
print("For least square:", "\n", "theta = ", thetaW, "\n", "NRMSE = ",  NRMSEW)
#####################################################
######################################################
# Validation set
N_val = 20000
U_val = np.linspace(0, 1, N_val)
y_val = (4*(U_val-0.5)**2)
noise = np.random.normal(0, .1, y_val.shape) * 3*(U_val+0.1)
Y_val = y_val + noise
X_val = Regressor(U_val, n)
Y_pred_val = np.dot(X_val, theta)
Y_pred_valWLS = np.dot(X_val, thetaW)
error = Y_val - Y_pred_val
errorWLS = Y_val - Y_pred_valWLS
NRMSE_LS = np.sqrt((1 / N_val) * (np.dot(error.transpose(), error)))
NRMSE_WLS = np.sqrt((1 / N_val) * (np.dot(errorWLS.transpose(), errorWLS)))
print("The NRMSE for Least square on validation set is: ", NRMSE_LS)
print("The NRMSE for Weighted Least square on validation set is: ", NRMSE_WLS)



