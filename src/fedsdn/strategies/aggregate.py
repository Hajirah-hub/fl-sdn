"""Server aggregation rules on flattened client updates.

Every rule takes `updates` (K x P tensor, Delta_k = w_k - w_global) and returns the aggregated
update Delta (P,), applied as w_global + Delta (FedAdam applies its own server step).
"""

from __future__ import annotations

import torch


def fedavg(updates: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    p = n.double() / n.double().sum()
    return (p[:, None].to(updates.dtype) * updates).sum(0)


def fednova(updates: torch.Tensor, n: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Wang et al. 2020 with plain SGD: d_k = Delta_k / tau_k, Delta = tau_eff * sum_k p_k d_k,
    tau_eff = sum_k p_k tau_k. Equals FedAvg when all tau_k are equal."""
    p = (n.double() / n.double().sum()).to(updates.dtype)
    tau = tau.to(updates.dtype)
    tau_eff = (p * tau).sum()
    return tau_eff * (p[:, None] * updates / tau[:, None]).sum(0)


class FedAdam:
    """Reddi et al. 2021 server Adam on the FedAvg pseudo-gradient (no bias correction)."""

    def __init__(self, server_lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.99, eps: float = 1e-3):
        self.lr, self.b1, self.b2, self.eps = server_lr, beta1, beta2, eps
        self.m = self.v = None

    def step(self, updates: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        delta = fedavg(updates, n)
        if self.m is None:
            self.m, self.v = torch.zeros_like(delta), torch.zeros_like(delta)
        self.m = self.b1 * self.m + (1 - self.b1) * delta
        self.v = self.b2 * self.v + (1 - self.b2) * delta**2
        return self.lr * self.m / (self.v.sqrt() + self.eps)


def krum(updates: torch.Tensor, f: int) -> torch.Tensor:
    """Blanchard et al. 2017: pick the update with the smallest sum of squared distances to its
    K - f - 2 nearest neighbours."""
    K = updates.shape[0]
    n_nb = max(K - f - 2, 1)
    d2 = torch.cdist(updates.double(), updates.double()) ** 2
    d2.fill_diagonal_(float("inf"))
    scores = d2.sort(dim=1).values[:, :n_nb].sum(1)
    return updates[int(torch.argmin(scores))].clone()


def coordinate_median(updates: torch.Tensor) -> torch.Tensor:
    """Coordinate-wise median; for even K the mean of the two middle values."""
    s = updates.sort(dim=0).values
    K = s.shape[0]
    if K % 2:
        return s[K // 2].clone()
    return 0.5 * (s[K // 2 - 1] + s[K // 2])


def trimmed_mean(updates: torch.Tensor, f: int) -> torch.Tensor:
    """Coordinate-wise: drop the f largest and f smallest values, average the rest."""
    K = updates.shape[0]
    if K - 2 * f < 1:
        raise ValueError("trimmed mean needs K > 2f")
    return updates.sort(dim=0).values[f:K - f].mean(0)
