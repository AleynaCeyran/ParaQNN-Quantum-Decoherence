"""
Training objectives for ParaQNN.

This module defines a composite loss over:
(1) signal reconstruction (Truth stream),
(2) noise-proxy alignment (Falsity stream),
(3) contradiction regularization that discourages excessive overlap between streams.

The formulation is intentionally minimal and dataset-agnostic; each simulation pipeline
provides (signal_target, noise_proxy) consistent with its data-generation contract.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParaconsistentLoss(nn.Module):
    """
    Composite objective for ParaQNN training.

    Terms:
        L_signal: MSE(signal_pred, signal_target)
        L_noise:  MSE(noise_pred, noise_proxy)
        L_contra: mean(ReLU((signal_pred + noise_pred) - 1)^2)

    Notes:
        - noise_proxy is an explicit supervision signal (e.g., |noisy - ideal|) produced by
          the dataset pipeline; it is not assumed to be a calibrated physical noise parameter.
    """

    def __init__(
        self,
        lambda_signal: float = 1.0,
        lambda_noise: float = 0.5,
        lambda_contradiction: float = 0.5,
    ):
        super().__init__()
        self.lambda_signal = float(lambda_signal)
        self.lambda_noise = float(lambda_noise)
        self.lambda_contradiction = float(lambda_contradiction)

        if self.lambda_signal < 0 or self.lambda_noise < 0 or self.lambda_contradiction < 0:
            raise ValueError("Loss weights must be non-negative.")

    def forward(
        self,
        signal_pred: torch.Tensor,
        noise_pred: torch.Tensor,
        signal_target: torch.Tensor,
        noise_proxy: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        signal_pred = signal_pred.float()
        noise_pred = noise_pred.float()
        signal_target = signal_target.float()
        noise_proxy = noise_proxy.float()

        l_signal = F.mse_loss(signal_pred, signal_target)
        l_noise = F.mse_loss(noise_pred, noise_proxy)

        violation = torch.relu((signal_pred + noise_pred) - 1.0)
        l_contradiction = torch.mean(violation * violation)

        total = (
            self.lambda_signal * l_signal
            + self.lambda_noise * l_noise
            + self.lambda_contradiction * l_contradiction
        )

        metrics = {
            "loss/total": float(total.detach().cpu().item()),
            "loss/signal": float(l_signal.detach().cpu().item()),
            "loss/noise_proxy": float(l_noise.detach().cpu().item()),
            "loss/contradiction": float(l_contradiction.detach().cpu().item()),
            "loss/weights_lambda_signal": float(self.lambda_signal),
            "loss/weights_lambda_noise": float(self.lambda_noise),
            "loss/weights_lambda_contradiction": float(self.lambda_contradiction),
        }
        return total, metrics

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"lambda_signal={self.lambda_signal}, "
            f"lambda_noise={self.lambda_noise}, "
            f"lambda_contradiction={self.lambda_contradiction})"
        )

