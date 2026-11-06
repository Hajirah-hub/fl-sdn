"""Essential aggregation tests."""

import torch

from torch.nn.utils import parameters_to_vector

from fedsdn.models import MLP
from fedsdn.sim import RunConfig, is_collapsed, local_train
from fedsdn.strategies.aggregate import (
    Scaffold, coordinate_median, fedavg, fednova, krum, trimmed_mean,
)


def test_fednova_equals_fedavg_with_equal_local_steps():
    g = torch.Generator().manual_seed(0)
    U = torch.randn(8, 50, generator=g)
    n = torch.tensor([100, 250, 300, 120, 900, 410, 150, 600], dtype=torch.float64)
    tau = torch.full((8,), 7.0, dtype=torch.float64)
    torch.testing.assert_close(fednova(U, n, tau), fedavg(U, n))


def test_fednova_differs_with_unequal_local_steps():
    U = torch.tensor([[1.0], [1.0]])
    n = torch.tensor([1.0, 1.0])
    # d = [1/1, 1/3]; tau_eff = 2 -> 2 * mean(1, 1/3) = 4/3; FedAvg -> 1
    torch.testing.assert_close(fednova(U, n, torch.tensor([1.0, 3.0])), torch.tensor([4.0 / 3.0]))


def test_median_and_trimmed_mean_known_vectors():
    U = torch.tensor([
        [1.0, 10.0, -5.0],
        [2.0, 20.0, 0.0],
        [3.0, 30.0, 5.0],
        [4.0, 40.0, 100.0],
        [100.0, -1000.0, 7.0],
        [5.0, 50.0, 6.0],
    ])
    # sorted columns: [1,2,3,4,5,100] [-1000,10,20,30,40,50] [-5,0,5,6,7,100]
    torch.testing.assert_close(coordinate_median(U), torch.tensor([3.5, 25.0, 5.5]))
    torch.testing.assert_close(coordinate_median(U[:5]), torch.tensor([3.0, 20.0, 5.0]))
    torch.testing.assert_close(trimmed_mean(U, 1), torch.tensor([3.5, 25.0, 4.5]))
    torch.testing.assert_close(trimmed_mean(U, 2), torch.tensor([3.5, 25.0, 5.5]))


def test_krum_rejects_outlier():
    U = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [0.1, 0.1], [0.05, 0.05], [50.0, 50.0]])
    out = krum(U, f=1)
    assert out.abs().max() < 1.0


def _toy_federation(strategy: str, freeze: bool = False, R: int = 3):
    """K = 3 clients with different sizes on synthetic data; returns the final global weights."""
    g = torch.Generator().manual_seed(0)
    X = torch.randn(300, 6, generator=g)
    y = (X[:, 0] > 0).long() + (X[:, 1] > 0.5).long()
    shards = [torch.arange(0, 60), torch.arange(60, 160), torch.arange(160, 300)]
    n = torch.tensor([len(s) for s in shards], dtype=torch.float64)
    cfg = RunConfig(kind="fed", lr=0.1, batch=32)
    cw = torch.ones(3)
    torch.manual_seed(0)
    model = MLP(6, 3)
    w = parameters_to_vector(model.parameters()).detach().clone()
    sc = Scaffold(3, w.numel(), freeze=freeze) if strategy == "scaffold" else None
    for r in range(R):
        U, taus = [], []
        for k, idx in enumerate(shards):
            torch.manual_seed(100 * r + k)
            gen = torch.Generator().manual_seed(100 * r + k)
            d, t = local_train(model, w, X[idx], y[idx], cw, cfg, gen, prox=False,
                               correction=sc.correction(k) if sc else None)
            U.append(d)
            taus.append(t)
        U, tau = torch.stack(U), torch.tensor(taus, dtype=torch.float64)
        w = w + (sc.step(U, n, tau, cfg.lr, [0, 1, 2]) if sc else fedavg(U, n))
    return w, sc


def test_scaffold_equals_fedavg_with_variates_frozen_at_zero():
    w_avg, _ = _toy_federation("fedavg")
    w_sc, sc = _toy_federation("scaffold", freeze=True)
    assert torch.equal(w_avg, w_sc)
    assert not sc.c.any() and not any(c.any() for c in sc.c_k.values())


def test_scaffold_variates_update_and_change_the_trajectory():
    w_avg, _ = _toy_federation("fedavg")
    w_sc, sc = _toy_federation("scaffold", freeze=False)
    assert not torch.allclose(w_avg, w_sc)
    # server variate stays the n_k-weighted mean of the client variates
    p = torch.tensor([60, 100, 140], dtype=torch.float32) / 300
    torch.testing.assert_close(sc.c, sum(p[k] * sc.c_k[k] for k in range(3)), atol=1e-5, rtol=1e-4)


def test_scaffold_client_variate_formula():
    # one client, c = c_k = 0: c_k+ = -Delta / (tau * lr)
    sc = Scaffold(1, 2)
    U = torch.tensor([[0.2, -0.4]])
    sc.step(U, torch.tensor([1.0]), torch.tensor([4.0]), 0.1, [0])
    torch.testing.assert_close(sc.c_k[0], torch.tensor([-0.5, 1.0]))
    torch.testing.assert_close(sc.c, torch.tensor([-0.5, 1.0]))


def test_collapse_flag():
    assert is_collapsed([1.0, 0.0, 0.0, 0.0, 0.0])          # everything predicted as Normal
    assert is_collapsed([0.0, 0.0, 0.995, 0.005, 0.0])
    assert not is_collapsed([1.0, 0.02, 0.0, 0.0, 0.0])
    assert not is_collapsed([0.97, 0.9, 0.95, 0.99, 0.8])
