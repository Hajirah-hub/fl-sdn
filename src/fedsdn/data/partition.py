"""Controller-level non-IID partitions (paper Sec. III-D; CLAUDE.md "Partition file format").

Partitions cover the training pool only (`split == "train"`) and are built on the partition
family: `family` mapped through the rare-class decision, so all rare families form one family
`Rare`. The same shards serve the multi-class and the binary task.

Schemes (K clients, F attack families):
  P0  uniform random: shuffle, equal shards.
  P1  per family c: q_c ~ Dir_K(alpha); the family's rows are dealt across clients by q_c.
  P2  s ~ Dir_K(beta); all rows shuffled and dealt by s (class mix = global mix in expectation).
  P3  benign dealt equally over all K; attack families round-robin, m_eff = max(m, ceil(F/K))
      per client (with replacement when K*m_eff > F); a family is split equally among holders.
  P4  combined skew: q_c ~ Dir_K(alpha) per family and one s ~ Dir_K(beta); family c is dealt
      by q~_c,k = q_c,k * s_k / sum_j q_c,j * s_j. Uniform s gives P1; uniform q gives P2 (in
      expectation). Shard sizes therefore depend on both draws; realized CV/JSD are recorded.

Dealing = shuffle the rows, then give consecutive chunks whose sizes come from the proportions
by largest-remainder rounding (sizes sum exactly to the row count).

Minimum shard rule: up to `max_attempts` fresh draws until every shard has >= n_min rows (shard =
the client's training-pool rows, validation included, so validation gets >= 10 rows). If no draw
succeeds, the attempt with the largest minimum shard is kept and random rows are moved from the
current largest shard to each deficient one.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fedsdn.data.preprocess import atomic_write

N_MIN = 100
MAX_ATTEMPTS = 100
VAL_FRAC = 0.10


# --------------------------------------------------------------------------------------------
# Settings and names
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Setting:
    scheme: str                   # P0 | P1 | P2 | P3 | P4
    alpha: float | None = None    # P1, P4
    beta: float | None = None     # P2, P4
    m: int | None = None          # P3

    @property
    def tag(self) -> str:
        if self.scheme == "P0":
            return "P0"
        if self.scheme == "P1":
            return f"P1_a{self.alpha:g}"
        if self.scheme == "P2":
            return f"P2_b{self.beta:g}"
        if self.scheme == "P3":
            return f"P3_m{self.m}"
        if self.scheme == "P4":
            return f"P4_a{self.alpha:g}b{self.beta:g}"
        raise ValueError(self.scheme)


SETTINGS = (
    Setting("P0"),
    *(Setting("P1", alpha=a) for a in (0.1, 0.5, 1.0, 10.0)),
    *(Setting("P2", beta=b) for b in (0.5, 1.0)),
    *(Setting("P3", m=m) for m in (1, 2)),
    Setting("P4", alpha=0.5, beta=0.5),
)


def partition_name(setting: Setting, K: int, seed: int) -> str:
    return f"{setting.tag}_K{K}_s{seed}"


def setting_rng(setting: Setting, seed: int) -> np.random.Generator:
    """Independent stream per (setting, seed); deterministic across runs and platforms."""
    return np.random.default_rng([seed, zlib.crc32(setting.tag.encode())])


# --------------------------------------------------------------------------------------------
# Training pool
# --------------------------------------------------------------------------------------------


def training_pool(flows: pd.DataFrame, class_of_family: dict[str, str]) -> pd.DataFrame:
    """Training-pool rows sorted by row_id with their partition family (rare families -> Rare)."""
    pool = flows.loc[flows["split"] == "train", ["row_id", "family"]]
    pool = pool.assign(pfamily=pool["family"].map(class_of_family))
    if pool["pfamily"].isna().any():
        raise ValueError("family missing from class_of_family")
    return pool.sort_values("row_id", kind="stable")[["row_id", "pfamily"]].reset_index(drop=True)


def family_order(names, benign: str) -> list[str]:
    rest = sorted(set(names) - {benign})
    return [benign] + rest if benign in set(names) else rest


# --------------------------------------------------------------------------------------------
# Dealing primitives and schemes
# --------------------------------------------------------------------------------------------


def split_counts(n: int, props) -> np.ndarray:
    """Integer sizes summing to n, proportional to props (largest remainder, ties -> lower index)."""
    props = np.asarray(props, dtype=float)
    raw = props / props.sum() * n
    base = np.floor(raw).astype(np.int64)
    order = np.argsort(-(raw - base), kind="stable")
    base[order[: n - int(base.sum())]] += 1
    return base


def _deal(client: np.ndarray, idx: np.ndarray, props, rng: np.random.Generator) -> None:
    perm = rng.permutation(idx)
    client[perm] = np.repeat(np.arange(len(props)), split_counts(len(idx), props))


def _dirichlet(rng: np.random.Generator, conc: float, K: int) -> np.ndarray:
    for _ in range(100):
        q = rng.dirichlet(np.full(K, conc))
        if np.all(np.isfinite(q)) and q.sum() > 0:
            return q
    raise RuntimeError("Dirichlet draw failed")


def assign_p0(fam, F, K, rng):
    client = np.empty(len(fam), dtype=np.int64)
    _deal(client, np.arange(len(fam)), np.ones(K), rng)
    return client, {}


def assign_p1(fam, F, K, rng, alpha):
    client = np.empty(len(fam), dtype=np.int64)
    q = {}
    for c in range(F):
        q[c] = _dirichlet(rng, alpha, K)
        _deal(client, np.flatnonzero(fam == c), q[c], rng)
    return client, {"q": {c: v.round(6).tolist() for c, v in q.items()}}


def assign_p2(fam, F, K, rng, beta):
    client = np.empty(len(fam), dtype=np.int64)
    s = _dirichlet(rng, beta, K)
    _deal(client, np.arange(len(fam)), s, rng)
    return client, {"s": s.round(6).tolist()}


def p4_proportions(q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Per-family label proportions q (K,) reweighted by quantity proportions s (K,)."""
    w = q * s
    total = w.sum()
    return w / total if total > 0 else s


def assign_p4(fam, F, K, rng, alpha, beta):
    client = np.empty(len(fam), dtype=np.int64)
    s = _dirichlet(rng, beta, K)
    q = {}
    for c in range(F):
        q[c] = _dirichlet(rng, alpha, K)
        _deal(client, np.flatnonzero(fam == c), p4_proportions(q[c], s), rng)
    return client, {"q": {c: v.round(6).tolist() for c, v in q.items()}, "s": s.round(6).tolist()}


def p3_holders(attack_codes: list[int], K: int, m: int, rng) -> tuple[int, dict[int, list[int]]]:
    """Round-robin over a seeded permutation of the attack families: client k holds the families
    at slots k*m_eff .. k*m_eff + m_eff - 1 (mod F). Covers every family since K*m_eff >= F."""
    F = len(attack_codes)
    m_eff = min(max(m, math.ceil(F / K)), F)
    order = [attack_codes[i] for i in rng.permutation(F)]
    holders: dict[int, list[int]] = {a: [] for a in attack_codes}
    for k in range(K):
        for j in range(m_eff):
            holders[order[(k * m_eff + j) % F]].append(k)
    return m_eff, holders


def assign_p3(fam, F, K, rng, m, benign_code):
    client = np.empty(len(fam), dtype=np.int64)
    if benign_code is not None:
        _deal(client, np.flatnonzero(fam == benign_code), np.ones(K), rng)
    attacks = [c for c in range(F) if c != benign_code]
    m_eff, holders = p3_holders(attacks, K, m, rng)
    for a, hs in holders.items():
        props = np.zeros(K)
        props[hs] = 1.0
        _deal(client, np.flatnonzero(fam == a), props, rng)
    return client, {"m_eff": m_eff, "holders": holders}


def _assign_once(setting: Setting, fam, F, K, rng, benign_code):
    if setting.scheme == "P0":
        return assign_p0(fam, F, K, rng)
    if setting.scheme == "P1":
        return assign_p1(fam, F, K, rng, setting.alpha)
    if setting.scheme == "P2":
        return assign_p2(fam, F, K, rng, setting.beta)
    if setting.scheme == "P3":
        return assign_p3(fam, F, K, rng, setting.m, benign_code)
    if setting.scheme == "P4":
        return assign_p4(fam, F, K, rng, setting.alpha, setting.beta)
    raise ValueError(setting.scheme)


# --------------------------------------------------------------------------------------------
# Minimum shard rule, validation hold-out
# --------------------------------------------------------------------------------------------


def move_rows_to_minimum(client: np.ndarray, K: int, n_min: int, rng) -> dict[int, int]:
    """Move random rows from the current largest shard to each shard below n_min (smallest
    first). Modifies `client` in place; returns rows received per client."""
    sizes = np.bincount(client, minlength=K)
    if sizes.sum() < K * n_min:
        raise ValueError(f"{sizes.sum()} rows cannot give {K} shards of {n_min}")
    received: dict[int, int] = {}
    for k in np.argsort(sizes, kind="stable"):
        while sizes[k] < n_min:
            big = int(np.argmax(sizes))
            d = int(min(n_min - sizes[k], sizes[big] - n_min))
            rows = rng.choice(np.flatnonzero(client == big), size=d, replace=False)
            client[rows] = k
            sizes[big] -= d
            sizes[k] += d
            received[int(k)] = received.get(int(k), 0) + d
    return received


def assign_with_minimum(setting, fam, F, K, rng, benign_code, n_min=N_MIN, max_attempts=MAX_ATTEMPTS):
    best = None
    for attempt in range(1, max_attempts + 1):
        client, extra = _assign_once(setting, fam, F, K, rng, benign_code)
        smallest = int(np.bincount(client, minlength=K).min())
        if smallest >= n_min:
            return client, extra, {"attempts": attempt, "rows_reassigned": False, "rows_moved": 0,
                                   "rows_moved_to_client": {}}
        if best is None or smallest > best[2]:
            best = (client, extra, smallest)
    client, extra, _ = best
    received = move_rows_to_minimum(client, K, n_min, rng)
    return client, extra, {"attempts": max_attempts, "rows_reassigned": True,
                           "rows_moved": int(sum(received.values())),
                           "rows_moved_to_client": {str(k): v for k, v in sorted(received.items())}}


def local_validation(client, fam, K, F, rng, frac=VAL_FRAC) -> np.ndarray:
    """Per client, round(frac * n_k) validation rows, allocated over families proportionally
    (largest remainder) and drawn at random within each family."""
    is_val = np.zeros(len(client), dtype=bool)
    for k in range(K):
        idx = np.flatnonzero(client == k)
        v = int(np.floor(frac * len(idx) + 0.5))
        if v == 0:
            continue
        fk = fam[idx]
        cnt = np.bincount(fk, minlength=F)
        per_f = split_counts(v, cnt)
        assert np.all(per_f <= cnt)
        for c in np.flatnonzero(per_f):
            is_val[rng.choice(idx[fk == c], size=int(per_f[c]), replace=False)] = True
    return is_val


# --------------------------------------------------------------------------------------------
# Heterogeneity statistics
# --------------------------------------------------------------------------------------------


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence, base 2 (in [0, 1])."""
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def mean_pairwise_jsd(counts: np.ndarray) -> float:
    """counts: K x F. Mean JSD over all client pairs of their label distributions."""
    dist = counts / counts.sum(axis=1, keepdims=True)
    pairs = list(combinations(range(len(dist)), 2))
    return float(np.mean([jsd(dist[a], dist[b]) for a, b in pairs]))


def size_cv(sizes) -> float:
    sizes = np.asarray(sizes, dtype=float)
    return float(sizes.std() / sizes.mean())


# --------------------------------------------------------------------------------------------
# Build and write
# --------------------------------------------------------------------------------------------


def make_partition(pool: pd.DataFrame, setting: Setting, K: int, seed: int, *, benign: str,
                   flows_sha256: str, dataset: str, n_min: int = N_MIN,
                   max_attempts: int = MAX_ATTEMPTS, val_frac: float = VAL_FRAC) -> tuple[pd.DataFrame, dict]:
    families = family_order(pool["pfamily"].unique(), benign)
    F = len(families)
    fam = pool["pfamily"].map({f: i for i, f in enumerate(families)}).to_numpy(np.int64)
    benign_code = families.index(benign) if benign in families else None
    rng = setting_rng(setting, seed)

    client, extra, shard_rule = assign_with_minimum(setting, fam, F, K, rng, benign_code,
                                                    n_min, max_attempts)
    is_val = local_validation(client, fam, K, F, rng, val_frac)

    counts = np.zeros((K, F), dtype=np.int64)
    np.add.at(counts, (client, fam), 1)
    val_counts = np.zeros((K, F), dtype=np.int64)
    np.add.at(val_counts, (client[is_val], fam[is_val]), 1)
    sizes = counts.sum(axis=1)

    sidecar = {
        "name": partition_name(setting, K, seed),
        "dataset": dataset,
        "scheme": setting.scheme,
        "params": {k: v for k, v in asdict(setting).items() if k != "scheme"},
        "K": K,
        "seed": seed,
        "flows_sha256": flows_sha256,
        "n_rows": int(len(pool)),
        "families": families,
        "n_min": n_min,
        "max_attempts": max_attempts,
        **shard_rule,
        "val_frac": val_frac,
        "shard_sizes": sizes.tolist(),
        "val_sizes": val_counts.sum(axis=1).tolist(),
        "client_family_counts": {str(k): dict(zip(families, map(int, counts[k]))) for k in range(K)},
        "client_val_family_counts": {str(k): dict(zip(families, map(int, val_counts[k])))
                                     for k in range(K)},
        "mean_pairwise_jsd": round(mean_pairwise_jsd(counts), 6),
        "jsd_note": "base-2 Jensen-Shannon divergence of per-client family distributions (whole shard)",
        "shard_size_cv": round(size_cv(sizes), 6),
        "min_shard": int(sizes.min()),
        "max_shard": int(sizes.max()),
    }
    if setting.scheme in ("P1", "P4"):
        sidecar["dirichlet_q"] = {families[c]: v for c, v in extra["q"].items()}
    if setting.scheme in ("P2", "P4"):
        sidecar["dirichlet_s"] = extra["s"]
    if setting.scheme == "P3":
        sidecar["m_eff"] = extra["m_eff"]
        sidecar["family_to_clients"] = {families[a]: hs for a, hs in extra["holders"].items()}
        if benign_code is not None:
            sidecar["family_to_clients"] = {benign: list(range(K)), **sidecar["family_to_clients"]}

    part = pd.DataFrame({
        "row_id": pool["row_id"].to_numpy(np.int64),
        "client_id": client.astype(np.int16),
        "is_val": is_val,
    })
    return part, sidecar


def write_partition(out_dir: Path, part: pd.DataFrame, sidecar: dict) -> Path:
    out_dir = Path(out_dir)
    path = out_dir / f"{sidecar['name']}.parquet"
    atomic_write(path, lambda t: part.to_parquet(t, index=False))
    atomic_write(path.with_suffix(".json"),
                 lambda t: Path(t).write_text(json.dumps(sidecar, indent=2), encoding="utf-8"))
    return path
