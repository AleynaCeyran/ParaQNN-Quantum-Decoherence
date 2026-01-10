"""
Paraconsistent Logic Primitives.

This module implements the core building blocks for ParaQNN's paraconsistent
reasoning capabilities. It defines the dual-channel layers and activations
that allow the network to process "Truth" (coherent signal) and "Falsity"
(noise/contradiction) as distinct but interacting streams.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DialetheistLayer(nn.Module):
    """
    A linear layer that couples Truth and Falsity channels.

    Unlike a standard linear layer, this layer maintains two streams and allows
    information to flow between them via cross-coupling weights.

    The transformation is defined as:
        z_t = W_tt * t_in + W_tf * f_in + b_t
        z_f = W_ff * f_in + W_ft * t_in + b_f

    Attributes:
        w_tt (nn.Linear): Weights mapping Truth to Truth (Signal preservation).
        w_ff (nn.Linear): Weights mapping Falsity to Falsity (Noise propagation).
        w_tf (nn.Linear): Weights mapping Falsity to Truth (Noise corrupting signal).
        w_ft (nn.Linear): Weights mapping Truth to Falsity (Signal generating noise).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize the DialetheistLayer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool): If set to False, the layer will not learn an additive bias.
                         Defaults to True.
        """
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)

        # Standard self-maps have bias
        self.w_tt = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.w_ff = nn.Linear(self.in_features, self.out_features, bias=bias)

        # Cross-coupling maps usually do not have bias to respect zero-interactions
        self.w_tf = nn.Linear(self.in_features, self.out_features, bias=False)
        self.w_ft = nn.Linear(self.in_features, self.out_features, bias=False)

    def forward(self, t_in: torch.Tensor, f_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            t_in (torch.Tensor): Input Truth tensor (batch_size, in_features).
            f_in (torch.Tensor): Input Falsity tensor (batch_size, in_features).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - z_t: Pre-activation Truth tensor (batch_size, out_features).
                - z_f: Pre-activation Falsity tensor (batch_size, out_features).
        """
        z_t = self.w_tt(t_in) + self.w_tf(f_in)
        z_f = self.w_ff(f_in) + self.w_ft(t_in)
        return z_t, z_f


class ParaconsistentActivation(nn.Module):
    """
    Non-linear activation function implementing paraconsistent negation.

    This activation models the interaction between Truth and Falsity.
    The Truth channel is attenuated by the presence of Falsity, modulated by
    a learnable parameter `alpha`. This implements the "damping" logic where
    strong noise (high Falsity) reduces confidence in the signal (Truth).

    Formulation:
        t_out = sigmoid(k * (z_t - alpha * z_f))
        f_out = sigmoid(k * z_f)

    Attributes:
        raw_alpha (nn.Parameter): Unconstrained parameter for alpha.
        k (float): Sharpness scaling factor.
    """

    def __init__(self, initial_alpha: float = 6.0, sharpness_k: float = 1.0):
        """
        Initialize the ParaconsistentActivation.

        Args:
            initial_alpha (float): Initial value for the damping parameter.
                                   Positive values mean Falsity suppresses Truth.
            sharpness_k (float): Scaling factor for the sigmoid input.
        """
        super().__init__()
        # Use a raw parameter and apply softplus in forward to ensure alpha > 0
        self.raw_alpha = nn.Parameter(torch.tensor(float(initial_alpha), dtype=torch.float32))
        self.k = float(sharpness_k)

    @property
    def alpha(self) -> torch.Tensor:
        """
        Return the rectified, positive alpha value.

        Returns:
            torch.Tensor: Softplus(raw_alpha).
        """
        return F.softplus(self.raw_alpha)

    def forward(self, z_t: torch.Tensor, z_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            z_t (torch.Tensor): Pre-activation Truth tensor.
            z_f (torch.Tensor): Pre-activation Falsity tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - t_out: Activated Truth tensor (in [0, 1]).
                - f_out: Activated Falsity tensor (in [0, 1]).
        """
        alpha = self.alpha
        t_out = torch.sigmoid(self.k * (z_t - alpha * z_f))
        f_out = torch.sigmoid(self.k * z_f)
        return t_out, f_out
