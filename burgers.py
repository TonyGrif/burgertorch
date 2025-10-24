import numpy as np
from pyDOE import lhs
import scipy


def main() -> None:
    # Global Space
    nu = 0.01 / np.pi                           # Predefined Kinematic Viscosity
    N_u = 100                                   # Number of training points
    N_f = 10000                                 # Number of collocation points
    layers = [2,20,20,20,20,20,20,20,1]         # Neural Network Layers

    print(f"Startng with globals: nu={nu}, N_u={N_u}, N_f={N_f}, layers={layers}")

    # Data Space
    data = scipy.io.loadmat("data/burgers_shock.mat")

    t = data["t"].flatten()[:,None]         # Time Component
    x = data["x"].flatten()[:,None]         # X Variable
    exact = np.real(data["usol"]).T         # U Component of the Solution

    X, T = np.meshgrid(x,t)         # Create Coordinate Grid of Points

    X_star = np.hstack((X.flatten()[:,None],T.flatten()[:,None]))       # Concat Mesh Matrixes Horizontally
    u_star = exact.flatten()[:,None]                                    # Needed for error metrics

    lower_bound = X_star.min(0)         # Minimum Row in the Stack
    upper_bound = X_star.max(0)         # Maxium Row in the Stack

    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))       # Aligning Values
    uu1 = exact[0:1,:].T

    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = exact[:,0:1]

    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = exact[:,-1:]

    X_u_train = np.vstack([xx1, xx2, xx3])                              # Training Data Matrix
    u_train = np.vstack([uu1, uu2, uu3])                                # Exact Matrix

    X_f_train = lower_bound + (upper_bound-lower_bound)*lhs(2, N_f)     # Collocation Matrix
    X_f_train = np.vstack((X_f_train, X_u_train))

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)      # Limit Training Points
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]

    # model = PINN(X_u_train, u_train, X_f_train, layers, lower_bound, upper_bound, nu)


if __name__ == "__main__":
    main()
