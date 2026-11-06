"""Reduced study: plain-PyTorch federated simulation (no Flower/Ray), all clients sequential.

Multi-class task only: rows with y_multi = -1 are dropped from training and evaluation (same as
masking them). Standardization from client count/sum/sumsq (scaling.py, validation excluded);
class weights n_train / (C * n_c) from the clients' training counts (validation excluded).

Every round logs the test metrics and the global validation macro-F1 (pooled counts over all
clients' validation sets). Per run three test numbers are reported: final round, val-selected
(round with the best global validation macro-F1, ties -> earliest; the test set is never used
for selection) and AUC (mean test macro-F1 over rounds 1..R). Centralized runs hold out a
stratified 10% of the training pool as their validation set.

Protocols: "chrono" (primary; processed/<ds>, partitions/<ds>) and "random" (secondary random
stratified 80/20 split; processed/<ds>_random, partitions/<ds>_random).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from fedsdn.data.scaling import aggregate_feature_stats, client_feature_stats
from fedsdn.models import MLP
from fedsdn.strategies import aggregate as agg

STRATEGIES = ("fedavg", "fedprox", "fednova", "fedadam", "krum", "median", "trimmed_mean")
PROTOCOL_SUFFIX = {"chrono": "", "random": "_random"}


@dataclass(frozen=True)
class RunConfig:
    kind: str                    # fed | central
    strategy: str = "central"
    partition: str | None = None # setting tag, e.g. P1_a0.5
    seed: int = 0
    protocol: str = "chrono"     # chrono | random
    attack: str | None = None    # sign_flip | targeted
    f: int = 0                   # malicious clients
    K: int = 8
    R: int = 30
    E: int = 1
    batch: int = 256
    lr: float = 0.05
    mu: float = 0.01             # FedProx
    f_hat: int = 2               # robust rules
    server_lr: float = 0.01      # FedAdam
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-3

    @property
    def run_id(self) -> str:
        if self.kind == "central":
            return f"{self.protocol}_central_s{self.seed}"
        rid = f"{self.protocol}_{self.strategy}_{self.partition}"
        if self.attack:
            rid += f"_{self.attack}_f{self.f}"
        return f"{rid}_s{self.seed}"


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------


@lru_cache(maxsize=4)
def load_flows(processed_dir: str):
    df = pd.read_parquet(Path(processed_dir) / "flows.parquet")
    df = df[df["split"].isin(["train", "test"])].reset_index(drop=True)
    feats = [c for c in df.columns if c.startswith("f_")]
    return df[["row_id", "split", "y_multi"]].copy(), df[feats].to_numpy(np.float32), feats


def class_names(meta: dict) -> list[str]:
    idx = meta["class_index"]
    return sorted(idx, key=idx.get)


def class_weights(counts: np.ndarray) -> torch.Tensor:
    C = len(counts)
    return torch.tensor(counts.sum() / (C * np.maximum(counts, 1)), dtype=torch.float32)


# --------------------------------------------------------------------------------------------
# Training / evaluation
# --------------------------------------------------------------------------------------------


def local_train(model, w0, X, y, cw, cfg: RunConfig, gen, prox: bool) -> tuple[torch.Tensor, int]:
    vector_to_parameters(w0.clone(), model.parameters())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=0.0)
    steps = 0
    for _ in range(cfg.E):
        perm = torch.randperm(len(y), generator=gen)
        for i in range(0, len(y), cfg.batch):
            idx = perm[i:i + cfg.batch]
            loss = F.cross_entropy(model(X[idx]), y[idx], weight=cw)
            if prox:
                loss = loss + cfg.mu / 2 * (parameters_to_vector(model.parameters()) - w0).pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
    return parameters_to_vector(model.parameters()).detach() - w0, steps


@torch.no_grad()
def predict(model, X) -> np.ndarray:
    model.eval()
    return model(X).argmax(1).numpy()


def macro_f1(y, pred, labels) -> float:
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def test_metrics(y: np.ndarray, pred: np.ndarray, C: int, rare: int, names: list[str]) -> dict:
    rec = recall_score(y, pred, labels=list(range(C)), average=None, zero_division=0)
    benign, attack = y == 0, y != 0
    out = {
        "macro_f1": macro_f1(y, pred, list(range(C))),
        "accuracy": float((y == pred).mean()),
        "fpr": float((pred[benign] != 0).mean()),
        "attack_fnr": float((pred[attack] == 0).mean()),
        "rare_recall": float(rec[rare]),
    }
    out.update({f"recall_{n}": float(r) for n, r in zip(names, rec)})
    return out


def select_round(val_f1: list[float]) -> int:
    """Index (0-based) of the best validation macro-F1; ties -> earliest round."""
    return int(np.argmax(np.asarray(val_f1)))


# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------


def run(cfg: RunConfig, processed_dir: str, partitions_dir: str, meta: dict) -> tuple[dict, pd.DataFrame]:
    t_start = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    info, X_raw, _ = load_flows(processed_dir)
    names = class_names(meta)
    C = len(names)
    labels = list(range(C))
    y_all = info["y_multi"].to_numpy()
    is_train = (info["split"] == "train").to_numpy()
    test_idx = np.flatnonzero(~is_train & (y_all >= 0))

    if cfg.kind == "central":
        tr_all = np.flatnonzero(is_train)
        tr_all_ok = tr_all[y_all[tr_all] >= 0]
        tr, va = train_test_split(tr_all_ok, test_size=0.1, stratify=y_all[tr_all_ok],
                                  random_state=cfg.seed)
        va_set = set(va.tolist())
        stats = [client_feature_stats(X_raw[[r for r in tr_all if r not in va_set]])]
        clients_tr, clients_val = [np.sort(tr)], [np.sort(va)]
    else:
        part = pd.read_parquet(Path(partitions_dir) / f"{cfg.partition}_K{cfg.K}_s{cfg.seed}.parquet")
        pos = pd.Series(np.arange(len(info)), index=info["row_id"].to_numpy())
        rows = pos.loc[part["row_id"].to_numpy()].to_numpy()
        cid, is_val = part["client_id"].to_numpy(), part["is_val"].to_numpy()
        clients_all = [rows[(cid == k) & ~is_val] for k in range(cfg.K)]
        stats = [client_feature_stats(X_raw[r]) for r in clients_all]           # val excluded
        clients_tr = [r[y_all[r] >= 0] for r in clients_all]                     # mask y = -1
        clients_val = [v[y_all[v] >= 0] for v in (rows[(cid == k) & is_val] for k in range(cfg.K))]

    mean, std = aggregate_feature_stats(stats)
    X = torch.from_numpy(((X_raw - mean) / std).astype(np.float32))
    counts = np.sum([np.bincount(y_all[r], minlength=C) for r in clients_tr], axis=0)
    cw = class_weights(counts)
    rare = int(np.argmin(counts))
    y_t = torch.from_numpy(y_all.astype(np.int64))
    y_test, X_test = y_all[test_idx], X[test_idx]
    val_all = np.concatenate(clients_val)
    val_owner = np.concatenate([np.full(len(v), k) for k, v in enumerate(clients_val)])
    y_val, X_val = y_all[val_all], X[val_all]

    model = MLP(X.shape[1], C)
    w = parameters_to_vector(model.parameters()).detach().clone()
    n_params = w.numel()
    rows_out = []

    def record(r, wall):
        vector_to_parameters(w, model.parameters())
        m = test_metrics(y_test, predict(model, X_test), C, rare, names)
        pv = predict(model, X_val)
        per_client = [macro_f1(y_val[val_owner == k], pv[val_owner == k],
                               sorted(set(y_val[val_owner == k].tolist())))
                      for k in range(len(clients_val)) if (val_owner == k).any()]
        bytes_round = 0 if cfg.kind == "central" else n_params * 4 * 2 * cfg.K
        rows_out.append({"run_id": cfg.run_id, "round": r, **m,
                         "val_macro_f1": macro_f1(y_val, pv, labels),
                         "worst_client_val_f1": float(min(per_client)),
                         "bytes": bytes_round, "wall_s": wall})

    malicious: list[int] = []
    if cfg.kind == "central":
        gen = torch.Generator().manual_seed(cfg.seed)
        tr_t = torch.from_numpy(clients_tr[0])
        for ep in range(1, cfg.R + 1):
            t0 = time.perf_counter()
            delta, _ = local_train(model, w, X[tr_t], y_t[tr_t], cw, cfg, gen, prox=False)
            w = w + delta
            record(ep, time.perf_counter() - t0)
    else:
        mal_rng = np.random.default_rng([cfg.seed, 2024])
        malicious = sorted(mal_rng.choice(cfg.K, size=cfg.f, replace=False).tolist()) if cfg.f else []
        n_k = torch.tensor([len(r) for r in clients_tr], dtype=torch.float64)
        tr_t = [torch.from_numpy(r) for r in clients_tr]
        y_local = []
        for k in range(cfg.K):
            yk = y_t[tr_t[k]].clone()
            if cfg.attack == "targeted" and k in malicious:
                yk[yk != 0] = 0                                   # relabel attacks as Normal
            y_local.append(yk)
        server_opt = agg.FedAdam(cfg.server_lr, cfg.beta1, cfg.beta2, cfg.eps) if cfg.strategy == "fedadam" else None
        for r in range(1, cfg.R + 1):
            t0 = time.perf_counter()
            updates, taus = [], []
            for k in range(cfg.K):
                gen = torch.Generator().manual_seed(cfg.seed * 1_000_003 + r * 101 + k)
                d, steps = local_train(model, w, X[tr_t[k]], y_local[k], cw, cfg, gen,
                                       prox=cfg.strategy == "fedprox")
                if cfg.attack == "sign_flip" and k in malicious:
                    d = -d
                updates.append(d)
                taus.append(steps)
            U = torch.stack(updates)
            tau = torch.tensor(taus, dtype=torch.float64)
            if cfg.strategy in ("fedavg", "fedprox"):
                delta = agg.fedavg(U, n_k)
            elif cfg.strategy == "fednova":
                delta = agg.fednova(U, n_k, tau)
            elif cfg.strategy == "fedadam":
                delta = server_opt.step(U, n_k)
            elif cfg.strategy == "krum":
                delta = agg.krum(U, cfg.f_hat)
            elif cfg.strategy == "median":
                delta = agg.coordinate_median(U)
            elif cfg.strategy == "trimmed_mean":
                delta = agg.trimmed_mean(U, cfg.f_hat)
            else:
                raise ValueError(cfg.strategy)
            w = w + delta
            record(r, time.perf_counter() - t0)

    rounds = pd.DataFrame(rows_out)
    sel = select_round(rounds["val_macro_f1"].tolist())
    metric_cols = [c for c in rounds.columns if c not in ("run_id", "round", "bytes", "wall_s")]
    final, best = rounds.iloc[-1], rounds.iloc[sel]
    summary = {
        "run_id": cfg.run_id, "protocol": cfg.protocol, "kind": cfg.kind, "strategy": cfg.strategy,
        "partition": cfg.partition or "central", "attack": cfg.attack or "none", "f": cfg.f,
        "seed": cfg.seed, "R": cfg.R, "rare_class": names[rare],
        "sel_round": int(best["round"]),
        **{f"sel_{k}": float(best[k]) for k in metric_cols},
        **{f"final_{k}": float(final[k]) for k in metric_cols},
        "auc_macro_f1": float(rounds["macro_f1"].mean()),
        "bytes_per_round": int(final["bytes"]), "n_params": n_params,
        "malicious": " ".join(map(str, malicious)),
        "wall_s": round(time.perf_counter() - t_start, 2),
    }
    return summary, rounds
