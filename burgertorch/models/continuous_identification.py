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

        # Initialize learnable PDE coefficients:
        # lambda_1 is a free parameter (initial 0.0)
        # lambda_2_param stores the log-like parameter so final lambda_2 = exp(lambda_2_param),
        # matching the TensorFlow original where lambda_2 = exp(self.lambda_2)
        self.lambda_1 = nn.Parameter(
            torch.tensor([0.0], dtype=torch.float32, device=self.device)
        )
        self.lambda_2_param = nn.Parameter(
            torch.tensor([-6.0], dtype=torch.float32, device=self.device)
        )

        # Optimizers: Adam for initial training, LBFGS for final refinement
        # We'll construct optimizers externally when training begins to ensure correct parameter sets.
        # However, prepare placeholders here:
        self.optimizer_adam = None
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
        """
        Compute the PDE `f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx`

        Args:
            X: X tensor
            t: Time tensor

        Returns:
            Torch Tensor

        lambda_2 = exp(lambda_2_param) to enforce positivity (same as TF).
        Autograd is used to compute derivatives w.r.t. x and t.
        """
        # make sure the inputs require gradients for derivative computation
        x = x.clone().detach().requires_grad_(True)
        t = t.clone().detach().requires_grad_(True)

        u = self.forward(x, t)  # u has computational graph from parameters & inputs

        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        # physical parameters
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2_param)

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx
        return f

    def loss_func(self):
        """
        Compute total loss:
            loss = MSE(u - u_pred) + MSE(f_pred)
        Mirrors: tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + tf.reduce_mean(tf.square(self.f_pred))
        """
        u_pred = self.forward(self.x, self.t)
        f_pred = self.net_f(self.x, self.t)
        mse_u = torch.mean((self.u - u_pred) ** 2)
        mse_f = torch.mean(f_pred**2)
        return mse_u + mse_f

    def train(self, nIter=10000, lr_adam=1e-3):
        """
        Train the PINN:
         - Run Adam for nIter iterations (printing progress every 10 iters),
         - Then run L-BFGS for final refinement (mimics tf.contrib.scipy L-BFGS-B behavior).
        """
        # set model to train mode
        super().train()

        # --- Adam optimizer setup ---
        # include both network parameters and PDE coefficient parameters
        params = list(self.model.parameters()) + [self.lambda_1, self.lambda_2_param]
        self.optimizer_adam = optim.Adam(params, lr=lr_adam)

        start_time = time.time()

        # Adam training loop (like TF train_op_Adam)
        for it in range(nIter):
            self.optimizer_adam.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer_adam.step()

            # print progress every 10 iterations (matching TF print frequency)
            if it % 10 == 0:
                elapsed = time.time() - start_time
                # compute readable values for lambdas exactly as TF did
                lambda_1_value = self.lambda_1.detach().cpu().numpy().ravel()[0]
                lambda_2_value = (
                    torch.exp(self.lambda_2_param).detach().cpu().numpy().ravel()[0]
                )
                print(
                    "It: %d, Loss: %.3e, Lambda_1: %.6f, Lambda_2: %.8f, Time: %.2f"
                    % (it, loss.item(), lambda_1_value, lambda_2_value, elapsed)
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
        final_lambda_2 = (
            torch.exp(self.lambda_2_param).detach().cpu().numpy().ravel()[0]
        )
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
