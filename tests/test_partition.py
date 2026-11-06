"""Partitioning and federated scaling on tiny synthetic data."""

import json

import numpy as np
import pandas as pd
import pytest

from fedsdn.data.partition import (
    Setting, make_partition, move_rows_to_minimum, p3_holders, p4_proportions, partition_name,
    split_counts,
    training_pool, write_partition,
)
from fedsdn.data.scaling import aggregate_feature_stats, client_feature_stats, standardize

FAMS = {"Normal": 3000, "DoS": 2500, "Probe": 2000, "BFA": 400, "DDoS": 300, "Rare": 120}


def make_pool(counts=FAMS, seed=0):
    rng = np.random.default_rng(seed)
    fam = np.concatenate([[f] * n for f, n in counts.items()])
    rng.shuffle(fam)
    return pd.DataFrame({"row_id": np.arange(len(fam), dtype=np.int64) * 3 + 7,
                         "pfamily": fam})


def build(setting, K=8, seed=0, pool=None, **kw):
    pool = make_pool() if pool is None else pool
    return make_partition(pool, setting, K, seed, benign="Normal", flows_sha256="x" * 64,
                          dataset="syn", **kw)


def shard_sizes(part, K):
    return np.bincount(part["client_id"], minlength=K)


# --- helpers ------------------------------------------------------------------------------------

def test_split_counts_sums_exactly():
    assert split_counts(10, [1, 1, 1]).tolist() == [4, 3, 3]
    assert split_counts(7, [0.0, 1.0, 0.0]).tolist() == [0, 7, 0]
    c = split_counts(1001, np.random.default_rng(0).dirichlet(np.ones(8) * 0.1))
    assert c.sum() == 1001 and (c >= 0).all()


def test_training_pool_maps_rare_families_and_keeps_train_only():
    flows = pd.DataFrame({
        "row_id": [5, 1, 2, 3, 4],
        "family": ["U2R", "Normal", "BOTNET", "DoS", "DoS"],
        "split": ["train", "train", "train", "test", "capped"],
    })
    pool = training_pool(flows, {"U2R": "Rare", "BOTNET": "Rare", "Normal": "Normal", "DoS": "DoS"})
    assert pool["row_id"].tolist() == [1, 2, 5]
    assert pool["pfamily"].tolist() == ["Normal", "Rare", "Rare"]


# --- every scheme: format, coverage of rows, determinism ----------------------------------------

@pytest.mark.parametrize("setting", [Setting("P0"), Setting("P1", alpha=0.5), Setting("P2", beta=0.5),
                                     Setting("P3", m=1), Setting("P3", m=2),
                                     Setting("P4", alpha=0.5, beta=0.5)])
def test_partition_covers_every_row_once_and_is_deterministic(setting):
    pool = make_pool()
    a, side_a = build(setting, pool=pool)
    b, side_b = build(setting, pool=pool)
    c, _ = build(setting, pool=pool, seed=1)
    assert a["row_id"].tolist() == pool["row_id"].tolist()
    assert a["client_id"].dtype == np.int16 and a["is_val"].dtype == bool
    assert a["client_id"].between(0, 7).all()
    pd.testing.assert_frame_equal(a, b)                       # same seed -> identical
    assert side_a == side_b
    assert not a["client_id"].equals(c["client_id"])         # other seed -> different
    assert sum(side_a["shard_sizes"]) == len(pool)
    assert min(side_a["shard_sizes"]) >= 100


def test_p4_proportions_reduce_to_p1_and_p2():
    rng = np.random.default_rng(0)
    q, s = rng.dirichlet(np.ones(8) * 0.5), rng.dirichlet(np.ones(8) * 0.5)
    np.testing.assert_allclose(p4_proportions(q, np.full(8, 1 / 8)), q)    # uniform s -> P1
    np.testing.assert_allclose(p4_proportions(np.full(8, 1 / 8), s), s)    # uniform q -> P2
    w = p4_proportions(q, s)
    np.testing.assert_allclose(w, q * s / (q * s).sum())


def test_p4_combines_label_and_quantity_skew():
    def mean_stat(setting, key):
        return np.mean([build(setting, seed=sd)[1][key] for sd in range(5)])
    p4 = Setting("P4", alpha=0.5, beta=0.5)
    assert mean_stat(p4, "mean_pairwise_jsd") > 3 * mean_stat(Setting("P2", beta=0.5), "mean_pairwise_jsd")
    assert mean_stat(p4, "shard_size_cv") > mean_stat(Setting("P0"), "shard_size_cv") + 0.2
    side = build(p4)[1]
    assert len(side["dirichlet_s"]) == 8 and set(side["dirichlet_q"]) == set(FAMS)


# --- P0 -----------------------------------------------------------------------------------------

def test_p0_roughly_balanced():
    part, side = build(Setting("P0"))
    sizes = shard_sizes(part, 8)
    assert sizes.max() - sizes.min() <= 1
    assert side["shard_size_cv"] < 0.001
    assert side["mean_pairwise_jsd"] < 0.01                  # label mix close to global
    assert side["attempts"] == 1 and not side["rows_reassigned"]


# --- P1 -----------------------------------------------------------------------------------------

def test_p1_smaller_alpha_gives_higher_jsd():
    def mean_jsd(alpha):
        return np.mean([build(Setting("P1", alpha=alpha), seed=s)[1]["mean_pairwise_jsd"]
                        for s in range(5)])
    j01, j1, j10 = mean_jsd(0.1), mean_jsd(1.0), mean_jsd(10.0)
    assert j01 > j1 > j10


# --- P2 -----------------------------------------------------------------------------------------

def test_p2_quantity_skew_keeps_label_mix():
    side = build(Setting("P2", beta=0.5))[1]
    assert side["shard_size_cv"] > 0.2
    p0 = build(Setting("P0"))[1]
    assert side["mean_pairwise_jsd"] < 0.05 and p0["mean_pairwise_jsd"] < 0.05


# --- P3 -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("K,m,F,expected_m_eff", [(8, 1, 5, 1), (8, 2, 5, 2), (3, 1, 5, 2), (2, 1, 5, 3)])
def test_p3_holders_cover_every_family(K, m, F, expected_m_eff):
    m_eff, holders = p3_holders(list(range(F)), K, m, np.random.default_rng(0))
    assert m_eff == expected_m_eff
    assert all(len(h) >= 1 for h in holders.values())          # every family covered
    per_client = {k: [a for a, hs in holders.items() if k in hs] for k in range(K)}
    assert all(len(v) == m_eff for v in per_client.values())    # m_eff distinct families each


@pytest.mark.parametrize("m", [1, 2])
def test_p3_partition_exclusivity_and_even_split(m):
    part, side = build(Setting("P3", m=m))
    counts = side["client_family_counts"]
    holders = side["family_to_clients"]
    assert holders["Normal"] == list(range(8))
    normal = [counts[str(k)]["Normal"] for k in range(8)]
    assert max(normal) - min(normal) <= 1
    for fam, hs in holders.items():
        if fam == "Normal":
            continue
        held = [counts[str(k)][fam] for k in hs]
        assert max(held) - min(held) <= 1                       # split evenly among holders
        assert sum(held) == FAMS[fam]
        assert all(counts[str(k)][fam] == 0 for k in range(8) if k not in hs)
    for k in range(8):
        n_attack = sum(1 for f in FAMS if f != "Normal" and counts[str(k)][f] > 0)
        assert n_attack == side["m_eff"] == m


# --- minimum shard rule -------------------------------------------------------------------------

def test_minimum_shard_rule_moves_rows_when_retries_fail():
    pool = make_pool({"Normal": 900, "DoS": 600})                # 2 families, K = 8, tiny alpha
    part, side = build(Setting("P1", alpha=0.01), pool=pool)
    sizes = shard_sizes(part, 8)
    assert sizes.min() >= 100 and sizes.sum() == 1500
    assert side["attempts"] == 100 and side["rows_reassigned"]
    assert side["rows_moved"] == sum(side["rows_moved_to_client"].values()) > 0


def test_minimum_shard_rule_retry_succeeds_without_moving():
    pool = make_pool({"Normal": 1500, "DoS": 1500})
    side = build(Setting("P1", alpha=0.3), pool=pool, seed=3)[1]   # first draws fail the minimum
    assert 1 < side["attempts"] < 100
    assert not side["rows_reassigned"] and side["rows_moved"] == 0 and side["min_shard"] >= 100


def test_move_rows_to_minimum():
    client = np.array([0] * 500 + [1] * 30 + [2] * 70)
    received = move_rows_to_minimum(client, 3, 100, np.random.default_rng(0))
    assert np.bincount(client, minlength=3).tolist() == [400, 100, 100]
    assert received == {1: 70, 2: 30}
    with pytest.raises(ValueError):
        move_rows_to_minimum(np.zeros(250, dtype=int), 3, 100, np.random.default_rng(0))


# --- validation hold-out ------------------------------------------------------------------------

def test_validation_is_10pct_and_stratified():
    part, side = build(Setting("P1", alpha=1.0))
    for k in range(8):
        n = side["shard_sizes"][k]
        assert side["val_sizes"][k] == int(np.floor(0.1 * n + 0.5))
        for fam, c in side["client_family_counts"][str(k)].items():
            v = side["client_val_family_counts"][str(k)][fam]
            assert v <= c and abs(v - 0.1 * c) <= 1.0          # proportional, rounding only


# --- write --------------------------------------------------------------------------------------

def test_write_partition_files(tmp_path):
    part, side = build(Setting("P3", m=1), seed=3)
    assert side["name"] == partition_name(Setting("P3", m=1), 8, 3) == "P3_m1_K8_s3"
    path = write_partition(tmp_path, part, side)
    assert path.name == "P3_m1_K8_s3.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), part)
    side2 = json.loads(path.with_suffix(".json").read_text())
    for key in ("flows_sha256", "attempts", "rows_reassigned", "client_family_counts",
                "mean_pairwise_jsd", "shard_size_cv", "family_to_clients"):
        assert key in side2
    assert Setting("P1", alpha=1.0).tag == "P1_a1" and Setting("P4", alpha=0.5, beta=0.5).tag == "P4_a0.5b0.5"


# --- federated scaling --------------------------------------------------------------------------

def test_federated_scaling_matches_pooled_numpy():
    rng = np.random.default_rng(0)
    x = np.column_stack([
        rng.normal(5.0, 2.0, 1000),
        rng.normal(1e6, 3e4, 1000),            # large offset
        rng.exponential(1e8, 1000),             # heavy scale (like Flow Byts/s)
        np.full(1000, 7.0),                     # constant
        np.zeros(1000),                         # constant zero
    ])
    shards = np.array_split(x, [100, 450, 460])          # uneven clients
    mean, std = aggregate_feature_stats([client_feature_stats(s) for s in shards])
    np.testing.assert_allclose(mean, x.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(std[:3], x[:, :3].std(axis=0), rtol=1e-9)
    assert std[3] == 1.0 and std[4] == 1.0 and mean[3] == 7.0
    z = standardize(x, mean, std)
    assert z.dtype == np.float32
    np.testing.assert_allclose(z[:, :3].mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(z[:, 3:], 0)


def test_federated_scaling_validation_rows_excluded_by_caller():
    x = np.arange(20, dtype=float).reshape(10, 2)
    is_val = np.zeros(10, dtype=bool)
    is_val[[0, 9]] = True
    mean, std = aggregate_feature_stats([client_feature_stats(x[~is_val])])
    np.testing.assert_allclose(mean, x[~is_val].mean(axis=0))
    np.testing.assert_allclose(std, x[~is_val].std(axis=0))
