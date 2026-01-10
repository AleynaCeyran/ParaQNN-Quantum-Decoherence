"""
ParaQNN Core Model Architecture.

This module implements the Paraconsistent Quantum Neural Network (ParaQNN),
a neuro-symbolic architecture designed to discover the governing laws of
open quantum systems from noisy time-series data.

It operates by propagating two coupled evidential streams:
1. **Truth (T)**: Represents the coherent signal or "ideal" physics.
2. **Falsity (F)**: Represents contradictory evidence, noise, or decoherence.

The network uses a dual-channel structure where these streams interact via
paraconsistent logic gates, allowing the model to separate signal from noise
without explicit supervision on the noise process.

Classes:
    ParaQNN: The main neural network module.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# Relative import within the package structure
from .dialetheist import DialetheistLayer, ParaconsistentActivation


class ParaQNN(nn.Module):
    """
    Dual-channel paraconsistent network for time-to-observable mapping.

    ParaQNN maps a time input `t` to an observable expectation value `O(t)`
    (e.g., population P(|1>)) while simultaneously estimating a noise/contradiction
    measure.

    The architecture consists of a stack of `DialetheistLayer` blocks followed by
    `ParaconsistentActivation` functions.

    Attributes:
        input_dim (int): Input feature dimension (typically 1 for time).
        hidden_dim (int): Latent dimension of the paraconsistent evidence space.
        output_dim (int): Output dimension (default 1 for a single bounded observable).
        num_layers (int): Total number of reasoning blocks (>= 2).
        f_init (str): Initialization strategy for the Falsity stream ("zeros" or "x").
        layers (nn.ModuleList): List of linear transformation layers.
        activations (nn.ModuleList): List of paraconsistent activation functions.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 3,
        initial_alpha: float = 6.0,
        sharpness_k: float = 1.0,
        f_init: str = "zeros",
    ):
        """
        Initialize the ParaQNN model.

        Args:
            input_dim (int): Input feature dimension. Defaults to 1.
            hidden_dim (int): Width of hidden layers. Defaults to 128.
            output_dim (int): Output dimension. Defaults to 1.
            num_layers (int): Number of layers (depth). Must be >= 2. Defaults to 3.
            initial_alpha (float): Initial value for the contradiction damping parameter alpha.
                                   Higher values suppress Truth more strongly when Falsity is high.
                                   Defaults to 6.0.
            sharpness_k (float): Scaling factor for the sigmoid activations. Defaults to 1.0.
            f_init (str): Falsity initialization mode.
                          "zeros": Falsity starts at 0.
                          "x": Falsity starts as a copy of the input (time).
                          Defaults to "zeros".

        Raises:
            ValueError: If `num_layers` < 2 or `f_init` is invalid.
        """
        super().__init__()

        if num_layers < 2:
            raise ValueError(f"num_layers must be >= 2, got {num_layers}")
        if f_init not in {"zeros", "x"}:
            raise ValueError(f"f_init must be one of {{'zeros','x'}}, got {f_init!r}")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.f_init = f_init

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # Input layer
        self.layers.append(DialetheistLayer(self.input_dim, self.hidden_dim))
        self.activations.append(ParaconsistentActivation(initial_alpha, sharpness_k))

        # Hidden layers
        for _ in range(self.num_layers - 2):
            self.layers.append(DialetheistLayer(self.hidden_dim, self.hidden_dim))
            self.activations.append(ParaconsistentActivation(initial_alpha, sharpness_k))

        # Output layer
        self.layers.append(DialetheistLayer(self.hidden_dim, self.output_dim))
        self.activations.append(ParaconsistentActivation(initial_alpha, sharpness_k))

    def get_logic_parameter(self) -> float:
        """
        Return the learned alpha value from the first layer.

        Alpha is an internal control parameter that modulates how strongly
        contradictory evidence attenuates the Truth stream. Under a fixed data/noise
        model, alpha can correlate with the effective noise/decoherence structure.

        Returns:
            float: The current value of alpha in the first activation layer.
        """
        return float(self.activations[0].alpha.detach().cpu().item())

    def get_alpha_vector(self) -> torch.Tensor:
        """
        Return all alpha values across the network layers.

        Returns:
            torch.Tensor: A 1D tensor of length `num_layers` containing alpha values.
        """
        with torch.no_grad():
            alphas = [a.alpha.detach().flatten().cpu() for a in self.activations]
        return torch.stack([x.squeeze() for x in alphas], dim=0)

    def get_alpha_summary(self) -> Dict[str, float]:
        """
        Return summary statistics of alpha parameters across layers.

        Returns:
            Dict[str, float]: Dictionary with keys "min", "max", "mean", "std".
        """
        v = self.get_alpha_vector().float()
        return {
            "min": float(v.min().item()),
            "max": float(v.max().item()),
            "mean": float(v.mean().item()),
            "std": float(v.std(unbiased=False).item()),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the network.

        The input `x` drives the Truth channel. The Falsity channel is initialized
        based on `f_init` and evolves through cross-coupling with Truth.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - truth_out: Predicted signal/observable (batch_size, output_dim).
                - falsity_out: Predicted noise/contradiction proxy (batch_size, output_dim).
        """
        t = x
        f = torch.zeros_like(x) if self.f_init == "zeros" else x

        for layer, activation in zip(self.layers, self.activations):
            z_t, z_f = layer(t, f)
            t, f = activation(z_t, z_f)

        return t, f

    def __repr__(self) -> str:
        a0 = float(self.activations[0].alpha.detach().cpu().item())
        return (
            "ParaQNN("
            f"in={self.input_dim}, hidden={self.hidden_dim}, out={self.output_dim}, "
            f"depth={self.num_layers}, f_init={self.f_init!r}, alpha0={a0:.3g})"
        )
