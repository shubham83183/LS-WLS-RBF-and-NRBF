# importing the required module
import matplotlib.pyplot as plt
import numpy as np



# Regressor function will take U and return regressor X
## which has polynomial terms of U (1, U, U^2 ......U^n)


def Regressor(U, n):
    num_cols = U.shape
    X = np.zeros((num_cols[0], n+1))
    for i in range(n+1):
        x = U ** i
        X[:, i] = x.transpose()
    return X

# This function implements least square and returns parameters theta


def least_sqr(X, Y):
    s = X.transpose()
    a = np.linalg.inv(np.dot(s, X))
    b = np.dot(s, Y)
    theta = np.dot(a, b)
    Y_pred = np.dot(X, theta)
    error = Y - Y_pred
    NRMSE = np.sqrt((1 / N) * (np.dot(error.transpose(), error)))
    return theta, NRMSE


N = 20
n = 2
U = np.linspace(0, 1, N)
y = (4*(U-0.5)**2)
noise = np.random.normal(0, .1, y.shape)
Y = y + noise
X = Regressor(U, n)
theta, NRMSE = least_sqr(X, Y)
print("theta: ", theta, '\n', )
print("The NRMSE between the process and the model output is: ", NRMSE)
###############################################################################
## VALIDATION SET
N_val = 20000
U_val = np.linspace(0, 1, N_val)
y_val = (4*(U_val-0.5)**2)
noise = np.random.normal(0, .1, y_val.shape)
Y_val = y_val + noise
X_val = Regressor(U_val, n)
Y_pred_val = np.dot(X_val, theta)
error = Y_val - Y_pred_val
NRMSE = np.sqrt((1 / N) * (np.dot(error.transpose(), error)))
print("The NRMSE of the validation data is: ", NRMSE)
plt.plot(U_val, y_val, 'b')     ## Actual values plotted with blue line
plt.plot(U_val, Y_pred_val, 'k') ## Predicted values plotted with black line
plt.ylabel('Output Y')
plt.xlabel('Input U')
plt.show()

