import numpy as np
from pyDOE import lhs
import scipy
from scipy.interpolate import griddata
import burgertorch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


def main() -> None:
    # Global Space
    nu = 0.01 / np.pi  # Predefined Kinematic Viscosity
    N_u = 100  # Number of training points
    N_f = 10000  # Number of collocation points
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 1]  # Neural Network Layers

    print(f"Startng with globals: nu={nu}, N_u={N_u}, N_f={N_f}, layers={layers}")

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

    network = burgertorch.ContinuousInferenceNetwork(
        X_u_train, u_train, X_f_train, lower_bound, upper_bound, nu, layers
    )
    print(f"Created sequential model structure: {network.model}")

    # Model Training and Metrics
    network.train_model(10000)

    u_pred, _ = network.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f"Calculated of u={error_u}")

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")

    # Graphing
    figwidth = 390 * (1 / 72.27)
    figheight = figwidth * (np.sqrt(5.0) - 1.0) / 2.0

    fig = plt.figure(
        figsize=(figwidth, figheight),
    )
    ax = fig.add_subplot(111)
    ax.axis("off")

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        U_pred.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=(t.min(), t.max(), x.min(), x.max()),
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, "w-", linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, exact[25, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[25, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=10)
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, exact[50, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[50, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_title("$t = 0.50$", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, exact[75, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[75, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_title("$t = 0.75$", fontsize=10)

    plt.show()


if __name__ == "__main__":
    main()
