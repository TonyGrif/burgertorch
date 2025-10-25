"""This module holds the Neural Network for continuous inference of
Burgers' Equation
"""

from typing import List
import torch
import torch.nn as nn


class ContinuousInferenceNetwork(nn.Module):
    """The Neural Network for continuous inference of Burgers' Equation"""

    def __init__(self, X_u, u, X_f, lb, ub, nu, layers) -> None:
        """Constructor for the ContinuousInferenceNetwork

        Args:
            X_u: The training data for this network
            u: The exact data
            X_f: The collocation points for this network
            lb: Lower bound of data
            ub: Upper bound of data
            nu: The viscosity
        """
        super(ContinuousInferenceNetwork, self).__init__()

        # Bound tensors
        self.lower_bound = torch.tensor(lb, dtype=torch.float32)
        self.upper_bound = torch.tensor(ub, dtype=torch.float32)

        # Training data
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32)

        # Exact data
        self.u = torch.tensor(u, dtype=torch.float32)

        # Collocation data
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32)

        # Viscosity
        self.nu = nu

        # Build model architecture
        self.model = self.build_network(layers)


    def build_network(self, layers: List[int]) -> nn.Sequential:
        """Build the Sequential Network architecture

        Args:
            layers: A List of layers of ints

        Returns:
            Sequential model
        """
        lays = []

        for i in range(len(layers) - 2):
            lays.append(nn.Linear(layers[i], layers[i+1]))

            # TODO: look into these methods more
            nn.init.xavier_normal_(lays[-1].weight)
            nn.init.zeros_(lays[-1].bias)

            lays.append(nn.Tanh())

        # Final Output
        lays.append(nn.Linear(layers[-2], layers[-1]))
        nn.init.xavier_normal_(lays[-1].weight)
        nn.init.zeros_(lays[-1].bias)

        return nn.Sequential(*lays)
