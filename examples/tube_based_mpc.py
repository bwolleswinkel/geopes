"""Example of tube-based MPC for a linear system with additive noise.

"""

import numpy as np
import cvxpy as cvx

# FIXME: This is off-course temporary...
#
import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
#
from geopes import geopes as geo
from geopes.geopes import is_in

# ------ Parameters ------

# Set the system dynamics
A = np.array([[0, 1], [-2, 3]])
B = np.array([[0], [1]])
n_x, n_u = A.shape[0], B.shape[1]

# Set the prediction horizon
N = 10

# Set the simulation length
T = 100

# ------ Script ------


def mpc_controller(x_k):
    X_sdp, U_sdp = cvx.Variable((n_x, N + 1)), cvx.Variable((n_u, N))
    X_sdp[:, 0] = cvx.Parameter(x_k)
    cost, constr = [], []
    for t in range(N):
        constr += [X_sdp[:, t + 1] == A @ X_sdp[:, t] + B @ U_sdp[:, t]]
        constr += [is_in(X_sdp[:, t], X)]


# Create the noise polytope
W = geo.norm_to_poly(0.25, n=n_x, p='1')

# Create the input and state constraints
U, X = geo.verts_to_poly(np.array([[-1, 1]])), geo.bounds_to_poly(lb=-10, ub=10) ** n_x

# Create an MPC controller
X_k, U_k = np.zeros((n_x, T + 1)), np.zeros((n_u, T))
for k in range(T):
    pass