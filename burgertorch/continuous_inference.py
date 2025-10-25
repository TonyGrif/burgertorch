"""This module holds the Neural Network for continuous inference of
Burgers' Equation
"""

from typing import List, Tuple

import numpy as np
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

        # ADAM optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def build_network(self, layers: List[Tuple[int, int]]) -> nn.Sequential:
        """Build the Sequential Network architecture

        Args:
            layers: A List of layers of Tuples of Ints

        Returns:
            Sequential model
        """
        lays = []

        for i in range(len(layers) - 1):
            lays.append(nn.Linear(layers[i][0], layers[i][1]))

            # TODO: look into these methods more
            nn.init.xavier_normal_(lays[-1].weight)
            nn.init.zeros_(lays[-1].bias)

            lays.append(nn.Tanh())

        # Final Output
        lays.append(nn.Linear(layers[-1][0], layers[-1][1]))
        nn.init.xavier_normal_(lays[-1].weight)
        nn.init.zeros_(lays[-1].bias)

        return nn.Sequential(*lays)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Forward pass to calculate `u(x,t)`

        Args:
            x: X tensor
            t: Time tensor
        """
        X = torch.cat((x, t), dim=1)
        X_norm = (
            2.0 * (X - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0
        )
        return self.model(X_norm)

    def net_f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the PDE `f = u_t + u*u_x - nu*u_xx`

        Args:
            x: X tensor
            t: Time tensor

        Returns:
            Torch Tensor
        """
        x.requires_grad = True
        t.requires_grad = True

        u = self.forward(x, t)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_func(self) -> torch.Tensor:
        """Compute total loss of boundary loss and PDE loss

        Returns:
            Torch Tensor
        """
        u_pred = self.forward(self.x_u, self.t_u)
        f_pred = self.net_f(self.x_f, self.t_f)
        mse_u = torch.mean((self.u - u_pred) ** 2)
        mse_f = torch.mean(f_pred**2)
        return mse_u + mse_f

    def train_model(self, max_epochs: int = 10000):
        """Train model with ADAM optimizer

        Args:
            max_epochs: The maxiumn number of epochs to do
        """
        self.model.train()

        for epoch in range(max_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.3e}")

    def predict(self, X_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the trained model on new points

        Args:
            X_star: Numpy Array

        Returns:
            Tuple of numpy ndarray type, containing 2 values
        """
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32)
        self.model.eval()

        with torch.no_grad():
            u_pred = self.forward(x, t)

        x.requires_grad = True
        t.requires_grad = True
        f_pred = self.net_f(x, t)

        return u_pred.detach().numpy(), f_pred.detach().numpy()
