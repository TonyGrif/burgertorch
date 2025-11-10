import numpy as np
import torch
from scipy.interpolate import griddata

import burgertorch

# For reproducibility
np.random.seed(1234)
torch.manual_seed(1234)


def main() -> None:
    # Global Space
    nu = 0.01 / np.pi  # Predefined Kinematic Viscosity
    N_u = 100  # Number of training points
    N_f = 10000  # Number of collocation points
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
    print(f"Startng with globals: nu={nu}, N_u={N_u}, N_f={N_f}, layers={layers}")

    (
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
    ) = burgertorch.prepare_continuous_inference(N_u, N_f)

    network = burgertorch.ContinuousInferenceNetwork(
        X_u_train, u_train, X_f_train, lower_bound, upper_bound, nu, layers
    )
    print(f"Created sequential model structure: {network.model}")

    # Training
    network.train_model(10000)

    u_pred, _ = network.predict(X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print(f"Calculated of u={error_u}")

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method="cubic")
    burgertorch.plot_results("output", x, t, X, T, exact, U_pred, X_u_train, u_train)


if __name__ == "__main__":
    main()
