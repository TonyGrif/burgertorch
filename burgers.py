from burgertorch.utils import read_mat_data
import numpy as np


def main() -> None:
    # Global Space
    nu = 0.01 / np.pi                           # Predefined Kinematic Viscosity
    noise = 0.0                                 # Predefined Noise Level
    N_u = 100                                   # Number of training points
    N_f = 10000                                 # Number of collocation points
    layers = [2,20,20,20,20,20,20,20,1]         # Neural Network Layers

    print(f"Startng with globals: nu={nu}, noise={noise}, N_u={N_u}, N_f={N_f}, layers={layers}")

    data = read_mat_data("data/burgers_shock.mat")

if __name__ == "__main__":
    main()
