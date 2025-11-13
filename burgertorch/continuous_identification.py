"""This module holds the Neural Network for continuous identification of
Burgers' Equation
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ContinuousIdentificationNetwork(nn.Module):
    """Physics informed neural network for Burgers continuous identification
    problems

    Attributes:
        train: Train the model
        predict: Predict with the trained model
    """

    def __init__(
        self,
        X: np.ndarray,
        u: np.ndarray,
        layers: List[Tuple[int, int]],
        lb: np.ndarray,
        ub: np.ndarray,
        device: Optional[str] = None,
    ) -> None:
        """Constructor for ContinuousIdentificationNetwork

        Args:
            X: The training data for this network
            u: The exact data
            layers:  The network architecture layer
            lb: Lower bound of data
            ub: Upper bound of data
            device: "cpu" or "cuda", default to "cpu"
        """
        super(ContinuousIdentificationNetwork, self).__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Bound tensors
        self.lb = torch.tensor(lb, dtype=torch.float32, device=self.device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=self.device)

        # Training data
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=self.device)
        self.t = torch.tensor(X[:, 1:2], dtype=torch.float32, device=self.device)

        # Exact data
        self.u = torch.tensor(u, dtype=torch.float32, device=self.device)

        # Build model architecture
        self.model = self._build_network(layers).to(self.device)

        self.lambda_1 = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32, device=self.device)
        )
        self.lambda_2 = nn.Parameter(
            torch.tensor([-6.0], dtype=torch.float32, device=self.device)
        )

        self.optimizer_lbfgs = None

    def _build_network(self, layers: List[Tuple[int, int]]) -> nn.Sequential:
        """Build the Sequential Network architecture

        Args:
            layers: A List of layers of Tuples of Ints

        Returns:
            Sequential model
        """
        lays = []

        for i in range(len(layers) - 1):
            lays.append(nn.Linear(layers[i][0], layers[i][1]))

            nn.init.xavier_normal_(lays[-1].weight)
            nn.init.zeros_(lays[-1].bias)

            lays.append(nn.Tanh())

        lays.append(nn.Linear(layers[-1][0], layers[-1][1]))
        nn.init.xavier_normal_(lays[-1].weight)
        nn.init.zeros_(lays[-1].bias)

        return nn.Sequential(*lays)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """Forward pass to compute `u(x,t)`

        Args:
            x: X tensor
            t: Time tensor
        """
        X = torch.cat([x, t], dim=1)
        X_norm = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return self.model(X_norm)

    def net_f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the PDE `f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx`

        Args:
            X: X tensor
            t: Time tensor

        Returns:
            Torch Tensor
        """
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = self.forward(x, t)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        f = u_t + self.lambda_1 * u * u_x - torch.exp(self.lambda_2) * u_xx
        return f

    def loss_func(self) -> torch.Tensor:
        """Compute total loss of boundary loss and PDE loss

        Returns:
            Torch Tensor
        """
        u_pred = self.forward(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        mse_u = torch.mean((self.u - u_pred) ** 2)
        mse_f = torch.mean(f_pred**2)
        return mse_u + mse_f

    def train_model(self, max_epochs: int = 10000):
        """Train model with ADAM and LBFGS optimizers

        Args:
            max_epochs: The maxiumn number of epochs to do
        """
        self.model.train()

        params = list(self.model.parameters()) + [self.lambda_1, self.lambda_2]
        adam = optim.Adam(params, lr=1e-3)

        start_time = time.time()
        for epoch in range(max_epochs):
            adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            adam.step()

            if epoch % 1000 == 0:
                elapsed = time.time() - start_time
                lambda_1_value = self.lambda_1.detach().cpu().numpy().ravel()[0]
                lambda_2_value = (
                    torch.exp(self.lambda_2).detach().cpu().numpy().ravel()[0]
                )
                print(
                    f"It: {epoch}, Loss: {loss.item():3e}, Lambda_1: {lambda_1_value:.6f}, Lambda_2: {lambda_2_value:.8f}, Time: {elapsed:.2f}"
                )
                start_time = time.time()

        # --- L-BFGS refinement ---
        # PyTorch's LBFGS has different options, but we can set history_size similar to maxcor.
        # We also pass all parameters (network + PDE parameters).
        self.optimizer_lbfgs = optim.LBFGS(
            params, max_iter=50000, history_size=50, line_search_fn="strong_wolfe"
        )

        # Closure required by PyTorch LBFGS: recompute loss & grads
        def closure():
            self.optimizer_lbfgs.zero_grad()
            loss = self.loss_func()
            loss.backward()
            return loss

        print("Starting L-BFGS optimization (this may take a while)...")
        start_time = time.time()
        # run LBFGS until convergence (PyTorch will call the closure multiple times internally)
        self.optimizer_lbfgs.step(closure)
        elapsed = time.time() - start_time

        # Final callback-style print similar to TF's loss_callback
        final_loss = self.loss_func().item()
        final_lambda_1 = self.lambda_1.detach().cpu().numpy().ravel()[0]
        final_lambda_2 = torch.exp(self.lambda_2).detach().cpu().numpy().ravel()[0]
        print(
            "Loss: %e, l1: %.5f, l2: %.5f"
            % (final_loss, final_lambda_1, final_lambda_2)
        )
        print("L-BFGS finished in %.2f seconds" % (elapsed))

    def predict(self, X_star):
        """
        Predict u and f at new points X_star (Nx2 numpy array).
        For u: use torch.no_grad() for efficiency (no need for gradients).
        For f: enable requires_grad on inputs and compute autograd derivatives.
        Returns numpy arrays (u_pred, f_pred).
        """
        self.eval()  # set to eval mode

        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=self.device)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=self.device)

        # compute u prediction without tracking gradients (faster)
        with torch.no_grad():
            u_pred = self.forward(x, t).cpu().numpy()

        # compute f prediction (requires gradients wrt inputs)
        # ensure inputs require grad
        xg = x.clone().detach().requires_grad_(True)
        tg = t.clone().detach().requires_grad_(True)

        f_pred = self.net_f(xg, tg).detach().cpu().numpy()

        return u_pred, f_pred
