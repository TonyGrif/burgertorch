import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata

import burgertorch

np.random.seed(1234)
torch.manual_seed(1234)


def main() -> None:
    # Global Space
    nu = 0.01 / np.pi  # Perdefined Viscosity
    N_u = 2000  # Number of training points
    layers = [
        (2, 20),
        (20, 20),
        (20, 20),
        (20, 20),
        (20, 20),
        (20, 20),
        (20, 20),
        (20, 20),
        (20, 1),
    ]  # Neural Network Layers

    print(f"Startng with globals: nu={nu}, N_u={N_u}, layers={layers}")

    # Data Space
    data = scipy.io.loadmat("data/burgers_shock.mat")

    t = data["t"].flatten()[:, None]  # Time Component
    x = data["x"].flatten()[:, None]  # X Variable
    Exact = np.real(data["usol"]).T  # U Component of Solution
    print(
        f"Aquired data with following sizes: t={t.size}, x={x.size}, exact={Exact.size}"
    )

    X, T = np.meshgrid(x, t)  # Create Coordinate Grid of Points
    X_star = np.hstack(
        (X.flatten()[:, None], T.flatten()[:, None])
    )  # Concat Mesh Matrixes Horizontally
    print(f"Stacking grid arrays to create new size of {X_star.shape}")
    u_star = Exact.flatten()[:, None]  # Needed for error metrics

    lb = X_star.min(0)  # Minimum Row in Stack
    ub = X_star.max(0)  # Maximum Row in Stack
    print(f"Lower bound = {lb}, upper bound = {ub}")

    # Limit data for training
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    network = burgertorch.ContinuousIdentificationNetwork(
        X_u_train, u_train, layers, lb, ub
    )
    print(f"Created sequential model structure: {network.model}")

    # Train zero iterations (this mirrors the TF main which sometimes calls train(0))
    network.train(nIter=0)  # runs LBFGS only (Adam loop is zero-length)

    # Predict on full grid
    u_pred, _ = network.predict(X_star)

    # compute relative L2 error in u
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    print(f"Calculated error u (clean)={error_u}")

    # reconstruct grid for plotting (same as TF code)
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")

    # extract identified PDE parameters
    lambda_1_value = network.lambda_1.detach().cpu().numpy().ravel()[0]
    lambda_2_value = torch.exp(network.lambda_2_param).detach().cpu().numpy().ravel()[0]

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print("Error of l1: %.5f%%" % (error_lambda_1))
    print("Error of l2: %.5f%%" % (error_lambda_2))

    # ---------------------------
    # Now run the noisy-data experiment (1% noise)
    # ---------------------------
    noise = 0.01
    u_train_noisy = u_train + noise * np.std(u_train) * np.random.randn(*u_train.shape)

    network_noisy = burgertorch.ContinuousIdentificationNetwork(
        X_u_train, u_train_noisy, layers, lb, ub
    )
    network_noisy.train(nIter=10000)  # run Adam 10000 its then L-BFGS

    u_pred_noisy, f_pred_noisy = network_noisy.predict(X_star)

    lambda_1_value_noisy = network_noisy.lambda_1.detach().cpu().numpy().ravel()[0]
    lambda_2_value_noisy = (
        torch.exp(network_noisy.lambda_2_param).detach().cpu().numpy().ravel()[0]
    )

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

    print("Error of l1 (noisy): %.5f%%" % (error_lambda_1_noisy))
    print("Error of l2 (noisy): %.5f%%" % (error_lambda_2_noisy))

    # ---------------------------
    # Plotting portion (kept similar to original)
    # ---------------------------
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
    ax.plot(x, Exact[25, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[25, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=10)
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[50, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_title("$t = 0.50$", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], "b-", linewidth=2, label="Exact")
    ax.plot(x, U_pred[75, :], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.set_title("$t = 0.75$", fontsize=10)

    plt.savefig("continuous_identifiction.pdf")
    plt.show()


if __name__ == "__main__":
    main()
