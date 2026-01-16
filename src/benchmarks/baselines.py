"""
Quantum Dynamics Benchmarking Suite: Baseline Models

Description:
    Implements standard classical and deep learning baselines for comparative 
    analysis against ParaQNN. Includes wrappers for SciKit-Learn regressors, 
    and PyTorch implementations of PINNs and GANs.

Models:
    - Random Forest (RF)
    - XGBoost (XGB)
    - Physics-Informed Neural Networks (PINN)
    - Generative Adversarial Networks (GAN)

Reference:
    Ceyran et al., Benchmarking Protocols [Methods Section].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Callable, Any


class BaselineFactory:
    """Factory class to instantiate and train baseline models."""
    
    @staticmethod
    def train_classical(model_type: str, X_train: np.ndarray, y_train: np.ndarray, seed: int = 42) -> Any:
        """
        Train a classical ML regressor.

        Args:
            model_type: 'RF' or 'XGB'.
            X_train: Training inputs.
            y_train: Training targets.
            seed: Random seed.

        Returns:
            Trained model with .predict() method.
        """
        if model_type == 'RF':
            model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=seed)
        elif model_type == 'XGB':
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=seed)
        else:
            raise ValueError(f"Unknown classical model: {model_type}")
        
        model.fit(X_train, y_train.ravel())
        return model


class PINN(nn.Module):
    """Standard MLP architecture for Physics-Informed Learning."""
    def __init__(self, hidden_dim: int = 128):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Linear output for regression
        )
    
    def forward(self, t):
        return self.net(t)


def train_pinn(
    t_train: torch.Tensor, 
    y_train: torch.Tensor, 
    physics_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int = 500,
    lr: float = 0.002,
    device: torch.device = torch.device('cpu'),
    physics_weight: float = 0.1
) -> nn.Module:
    """
    Trains a PINN using a composite loss: L_data + lambda * L_physics.

    Args:
        t_train: Time inputs (requires_grad=True).
        y_train: Target values.
        physics_loss_fn: Function computing residual physics loss.
        epochs: Number of training epochs.
        lr: Learning rate.
        device: Torch device.
        physics_weight: Weighting factor for the physics loss term.
    """
    model = PINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        
        # 1. Data Fitting Loss
        u_pred = model(t_train)
        loss_data = torch.mean((u_pred - y_train)**2)
        
        # 2. Physics Residual Loss (Computed via Autograd)
        loss_physics = physics_loss_fn(u_pred, t_train)
        
        # Weighted Sum
        total_loss = loss_data + physics_weight * loss_physics
        
        total_loss.backward()
        optimizer.step()
        
    return model


class GANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)


class GANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)


def train_gan(
    t_train: torch.Tensor, 
    y_train: torch.Tensor, 
    epochs: int = 500,
    lr: float = 0.002,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """
    Trains a simple regression GAN for time-series reconstruction.

    Args:
        t_train: Inputs.
        y_train: Targets.
        epochs: Training epochs.
        lr: Learning rate.
        device: Torch device.
    """
    G = GANGenerator().to(device)
    D = GANDiscriminator().to(device)
    
    opt_G = optim.Adam(G.parameters(), lr=lr)
    opt_D = optim.Adam(D.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    G.train()
    for _ in range(epochs):
        # --- Train Discriminator ---
        opt_D.zero_grad()
        real_labels = torch.ones(len(y_train), 1, device=device)
        fake_labels = torch.zeros(len(y_train), 1, device=device)
        
        # Real
        out_real = D(y_train)
        loss_real = criterion(out_real, real_labels)
        
        # Fake
        fake_data = G(t_train)
        out_fake = D(fake_data.detach())
        loss_fake = criterion(out_fake, fake_labels)
        
        d_loss = loss_real + loss_fake
        d_loss.backward()
        opt_D.step()
        
        # --- Train Generator ---
        opt_G.zero_grad()
        fake_data = G(t_train)
        out_fake = D(fake_data)
        
        # Generator Loss: Fool Discriminator + L2 Reconstruction Constraint
        g_adv = criterion(out_fake, real_labels)
        g_l2 = torch.mean((fake_data - y_train)**2) # Guide G to match data distribution
        g_loss = 0.1 * g_adv + 0.9 * g_l2
        
        g_loss.backward()
        opt_G.step()
        
    return G
