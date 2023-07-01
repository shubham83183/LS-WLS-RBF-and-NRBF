import matplotlib.pyplot as plt
import numpy as np
from Weighted import *
from least_square import Regressor


N = 5
Mu1 = 0.3
Mu2 = 0.5
Mu3 = 0.7
sigma = 0.7
U = np.linspace(0, 1, N)
y = (4*(U-0.5)**2)
noise = np.random.normal(0, .1, y.shape)
Y = y + noise

g1 = np.exp(-0.5*((U-Mu1)/sigma)**2)
g2 = np.exp(-0.5*((U-Mu2)/sigma)**2)
g3 = np.exp(-0.5*((U-Mu3)/sigma)**2)
X = np.array([g1, g2, g3]).transpose()
N1 = np.array([g1/g1.sum(), g2/g2.sum(), g3/g3.sum()]).transpose()
theta, NRMSE = least_sqr(X, Y)
theta_norm, NRMSE_norm = least_sqr(N1, Y)
print("LS solution without normalisation", "\n", "theta: ", theta, '\n',"NRMSE: ", NRMSE)
print("Normalised LS solution", "\n", "theta: ", theta_norm, '\n', "NRMSE", NRMSE_norm)
####################################################################
print(N1)
print(N1[:, 0])