"""Federated standardization (paper Sec. III-C "Federated scaling").

Each client sends count, per-feature sum and per-feature sum of squares over its training rows
(validation rows excluded); the server combines them into a global mean and population std
(ddof = 0). No client ever reveals its rows.

Accumulation is in float64. var = E[x^2] - mean^2 loses relative precision of about
eps * E[x^2] / var, so a feature whose variance is below `rel_tol * E[x^2]` is treated as
constant (numerically indistinguishable from it) and gets std = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REL_TOL = 1e-12


@dataclass(frozen=True)
class FeatureStats:
    count: int
    sum: np.ndarray       # float64, shape (d,)
    sumsq: np.ndarray     # float64, shape (d,)


def client_feature_stats(x: np.ndarray) -> FeatureStats:
    """x: (n, d) training rows of one client."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be 2-D")
    return FeatureStats(count=int(x.shape[0]), sum=x.sum(axis=0), sumsq=np.square(x).sum(axis=0))


def aggregate_feature_stats(stats: list[FeatureStats], rel_tol: float = REL_TOL) -> tuple[np.ndarray, np.ndarray]:
    """Server side: global mean and std; constant features get std 1."""
    n = sum(s.count for s in stats)
    if n == 0:
        raise ValueError("no rows")
    total = np.sum([s.sum for s in stats], axis=0)
    total_sq = np.sum([s.sumsq for s in stats], axis=0)
    mean = total / n
    ex2 = total_sq / n
    var = np.maximum(ex2 - mean**2, 0.0)
    constant = var <= rel_tol * ex2
    std = np.where(constant, 1.0, np.sqrt(var))
    return mean, std


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float64) - mean) / std).astype(np.float32)
