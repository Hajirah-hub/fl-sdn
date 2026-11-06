"""Essential aggregation tests."""

import torch

from fedsdn.strategies.aggregate import coordinate_median, fedavg, fednova, krum, trimmed_mean


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
