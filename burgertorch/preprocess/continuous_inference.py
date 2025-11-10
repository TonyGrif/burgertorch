"""This modules contains code for preparing data for continuous inference"""

import numpy as np
import scipy
from pyDOE import lhs


def prepare_continuous_inference(N_u: int, N_f: int):
    """Prepare the data for the model"""
    # Data Space
    data = scipy.io.loadmat("data/burgers_shock.mat")

    t = data["t"].flatten()[:, None]  # Time Component
    x = data["x"].flatten()[:, None]  # X Variable
    exact = np.real(data["usol"]).T  # U Component of the Solution
    print(
        f"Aquired data with following sizes: t={t.size}, x={x.size}, exact={exact.size}"
    )

    X, T = np.meshgrid(x, t)  # Create Coordinate Grid of Points
    print(f"Created meshgrids with sizes: X={X.shape}, T={T.shape}")

    X_star = np.hstack(
        (X.flatten()[:, None], T.flatten()[:, None])
    )  # Concat Mesh Matrixes Horizontally
    print(f"Stacking grid arrays to create new size of {X_star.shape}")
    u_star = exact.flatten()[:, None]  # Needed for error metrics

    lower_bound = X_star.min(0)  # Minimum Row in the Stack
    upper_bound = X_star.max(0)  # Maxium Row in the Stack
    print(f"Lower bound = {lower_bound}, upper bound = {upper_bound}")

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))  # Aligning Values
    uu1 = exact[0:1, :].T

    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = exact[:, 0:1]

    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])  # Training Data Matrix
    print(f"Aquired training data of shape {X_u_train.shape}")
    u_train = np.vstack([uu1, uu2, uu3])  # Exact Matrix
    print(f"Aquired training exact training data of shape {u_train.shape}")

    X_f_train = lower_bound + (upper_bound - lower_bound) * lhs(
        2, N_f
    )  # Collocation Matrix
    X_f_train = np.vstack((X_f_train, X_u_train))
    print(f"Aquired collocation training data of shape {X_f_train.shape}")

    idx = np.random.choice(
        X_u_train.shape[0], N_u, replace=False
    )  # Limit Training Points
    X_u_train = X_u_train[idx, :]
    print(f"Limited training data to shape {X_u_train.shape}")
    u_train = u_train[idx, :]
    print(f"Limited exact training data to shape {u_train.shape}")

    return (
        x,
        t,
        X,
        T,
        exact,
        X_star,
        u_star,
        X_u_train,
        u_train,
        X_f_train,
        lower_bound,
        upper_bound,
    )
