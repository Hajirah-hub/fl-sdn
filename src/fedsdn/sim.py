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

STRATEGIES = ("fedavg", "fedprox", "fednova", "fedadam", "scaffold", "krum", "median", "trimmed_mean")
PROTOCOL_SUFFIX = {"chrono": "", "random": "_random"}


@dataclass(frozen=True)
class RunConfig:
    kind: str                    # fed | central | local (local-only baseline: each client alone)
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
    per_step_seed: bool = False  # re-seed torch (dropout) per (round, client); used to match Flower
    tag: str = ""                # appended to run_id (e.g. tuning trial "t03")

    @property
    def run_id(self) -> str:
        sfx = f"_{self.tag}" if self.tag else ""
        if self.kind == "central":
            return f"{self.protocol}_central_s{self.seed}{sfx}"
        if self.kind == "local":
            return f"{self.protocol}_local_{self.partition}_s{self.seed}{sfx}"
        rid = f"{self.protocol}_{self.strategy}_{self.partition}"
        if self.attack:
            rid += f"_{self.attack}_f{self.f}"
        return f"{rid}_s{self.seed}" + ("_ps" if self.per_step_seed else "") + sfx


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


def step_seed(seed: int, r: int, k: int) -> int:
    """Seed of client k's local data order in round r (also its dropout seed when per_step_seed)."""
    return seed * 1_000_003 + r * 101 + k


def local_train(model, w0, X, y, cw, cfg: RunConfig, gen, prox: bool,
                torch_seed: int | None = None,
                correction: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Plain SGD from w0 for E epochs; returns (w - w0, steps). `correction` (SCAFFOLD c - c_k) is
    added to the gradient before every step."""
    if torch_seed is not None:
        torch.manual_seed(torch_seed)          # dropout masks independent of earlier RNG use
    vector_to_parameters(w0.clone(), model.parameters())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.0, weight_decay=0.0)
    corr = None
    if correction is not None:
        sizes = [p.numel() for p in model.parameters()]
        corr = [c.view_as(p) for c, p in zip(correction.split(sizes), model.parameters())]
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
            if corr is not None:
                for p, c in zip(model.parameters(), corr):
                    p.grad.add_(c)
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


COLLAPSE_HI, COLLAPSE_LO = 0.99, 0.01


def is_collapsed(recalls) -> bool:
    """Collapse: the model predicts (nearly) one class for every test flow, i.e. one class has
    test recall >= 0.99 and every other class <= 0.01. Uses only per-class recalls, so it can be
    applied to logged rounds after the fact; a diverged (NaN) model also counts as collapsed."""
    r = np.sort(np.asarray(recalls, float))[::-1]
    return bool(r[0] >= COLLAPSE_HI and (r[1:] <= COLLAPSE_LO).all())


def select_round(val_f1: list[float]) -> int:
    """Index (0-based) of the best validation macro-F1; ties -> earliest round."""
    return int(np.argmax(np.asarray(val_f1)))


# --------------------------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------------------------


def run(cfg: RunConfig, processed_dir: str, partitions_dir: str, meta: dict) -> tuple[dict, pd.DataFrame]:
    if cfg.kind == "local":
        return run_local(cfg, processed_dir, partitions_dir, meta)
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
        pt = predict(model, X_test)
        m = test_metrics(y_test, pt, C, rare, names)
        m["top_pred_share"] = float(np.bincount(pt).max() / len(pt))      # share of the most predicted class
        pv = predict(model, X_val)
        per_client = [macro_f1(y_val[val_owner == k], pv[val_owner == k],
                               sorted(set(y_val[val_owner == k].tolist())))
                      for k in range(len(clients_val)) if (val_owner == k).any()]
        # SCAFFOLD sends (Delta, Delta c) up and (w, c) down: twice the FedAvg payload
        bytes_round = 0 if cfg.kind == "central" else \
            n_params * 4 * 2 * cfg.K * (2 if cfg.strategy == "scaffold" else 1)
        rows_out.append({"run_id": cfg.run_id, "round": r, **m,
                         "val_macro_f1": macro_f1(y_val, pv, labels),
                         "worst_client_val_f1": float(min(per_client)),
                         "bytes": bytes_round, "wall_s": wall})

    malicious: list[int] = []
    diverged = False
    if cfg.kind == "central":
        gen = torch.Generator().manual_seed(cfg.seed)
        tr_t = torch.from_numpy(clients_tr[0])
        for ep in range(1, cfg.R + 1):
            t0 = time.perf_counter()
            delta, _ = local_train(model, w, X[tr_t], y_t[tr_t], cw, cfg, gen, prox=False)
            w = w + delta
            diverged = diverged or not bool(torch.isfinite(w).all())
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
        scaffold = agg.Scaffold(cfg.K, n_params) if cfg.strategy == "scaffold" else None
        for r in range(1, cfg.R + 1):
            t0 = time.perf_counter()
            updates, taus = [], []
            for k in range(cfg.K):
                ss = step_seed(cfg.seed, r, k)
                gen = torch.Generator().manual_seed(ss)
                d, steps = local_train(model, w, X[tr_t[k]], y_local[k], cw, cfg, gen,
                                       prox=cfg.strategy == "fedprox",
                                       torch_seed=ss if cfg.per_step_seed else None,
                                       correction=scaffold.correction(k) if scaffold else None)
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
            elif cfg.strategy == "scaffold":
                delta = scaffold.step(U, n_k, tau, cfg.lr, list(range(cfg.K)))
            elif cfg.strategy == "krum":
                delta = agg.krum(U, cfg.f_hat)
            elif cfg.strategy == "median":
                delta = agg.coordinate_median(U)
            elif cfg.strategy == "trimmed_mean":
                delta = agg.trimmed_mean(U, cfg.f_hat)
            else:
                raise ValueError(cfg.strategy)
            w = w + delta
            diverged = diverged or not bool(torch.isfinite(w).all())
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
        "diverged": diverged,
        "collapsed": is_collapsed([final[f"recall_{n}"] for n in names]),
        "lr": cfg.lr, "mu": cfg.mu, "server_lr": cfg.server_lr, "eps": cfg.eps, "tag": cfg.tag,
        "wall_s": round(time.perf_counter() - t_start, 2),
    }
    return summary, rounds


def run_local(cfg: RunConfig, processed_dir: str, partitions_dir: str, meta: dict) -> tuple[dict, pd.DataFrame]:
    """Local-only baseline: every client trains alone on its own shard for R epochs (no
    communication). Standardization and class weights come from the client's own training rows;
    all clients start from the same initial model. Each client picks its epoch by its own
    validation macro-F1 (classes present in its validation set; ties -> earliest) and is scored
    on the global test set. Summary: mean / worst / best client test macro-F1."""
    t_start = time.perf_counter()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    info, X_raw, _ = load_flows(processed_dir)
    names = class_names(meta)
    C = len(names)
    y_all = info["y_multi"].to_numpy()
    is_train = (info["split"] == "train").to_numpy()
    test_idx = np.flatnonzero(~is_train & (y_all >= 0))
    y_test = y_all[test_idx]
    part = pd.read_parquet(Path(partitions_dir) / f"{cfg.partition}_K{cfg.K}_s{cfg.seed}.parquet")
    pos = pd.Series(np.arange(len(info)), index=info["row_id"].to_numpy())
    rows = pos.loc[part["row_id"].to_numpy()].to_numpy()
    cid, is_val = part["client_id"].to_numpy(), part["is_val"].to_numpy()
    y_t = torch.from_numpy(y_all.astype(np.int64))
    model = MLP(X_raw.shape[1], C)
    w_init = parameters_to_vector(model.parameters()).detach().clone()
    rows_out, clients = [], []
    for k in range(cfg.K):
        all_k = rows[(cid == k) & ~is_val]
        mean, std = aggregate_feature_stats([client_feature_stats(X_raw[all_k])])
        Xk = torch.from_numpy(((X_raw - mean) / std).astype(np.float32))
        tr = all_k[y_all[all_k] >= 0]
        va = rows[(cid == k) & is_val]
        va = va[y_all[va] >= 0]
        counts = np.bincount(y_all[tr], minlength=C)
        cw = class_weights(counts)
        va_labels = sorted(set(y_all[va].tolist()))
        tr_t = torch.from_numpy(tr)
        gen = torch.Generator().manual_seed(step_seed(cfg.seed, 0, k))
        w = w_init.clone()
        hist = []
        for ep in range(1, cfg.R + 1):
            delta, _ = local_train(model, w, Xk[tr_t], y_t[tr_t], cw, cfg, gen, prox=False)
            w = w + delta
            vector_to_parameters(w, model.parameters())
            pt = predict(model, Xk[test_idx])
            row = {"run_id": cfg.run_id, "client": k, "round": ep,
                   "macro_f1": macro_f1(y_test, pt, list(range(C))),
                   "accuracy": float((y_test == pt).mean()),
                   "val_macro_f1": macro_f1(y_all[va], predict(model, Xk[va]), va_labels)}
            hist.append(row)
        rows_out += hist
        h = pd.DataFrame(hist)
        best = h.iloc[select_round(h["val_macro_f1"].tolist())]
        clients.append({"client": k, "n_train": len(tr), "n_classes": int((counts > 0).sum()),
                        "sel_round": int(best["round"]), "sel_f1": float(best["macro_f1"]),
                        "final_f1": float(h["macro_f1"].iloc[-1])})
    cl = pd.DataFrame(clients)
    summary = {
        "run_id": cfg.run_id, "protocol": cfg.protocol, "kind": "local", "strategy": "local",
        "partition": cfg.partition, "attack": "none", "f": 0, "seed": cfg.seed, "R": cfg.R,
        "sel_macro_f1": float(cl["sel_f1"].mean()), "sel_worst_macro_f1": float(cl["sel_f1"].min()),
        "sel_best_macro_f1": float(cl["sel_f1"].max()),
        "final_macro_f1": float(cl["final_f1"].mean()), "final_worst_macro_f1": float(cl["final_f1"].min()),
        "final_best_macro_f1": float(cl["final_f1"].max()),
        "client_sel_f1": " ".join(f"{v:.4f}" for v in cl["sel_f1"]),
        "client_n_classes": " ".join(map(str, cl["n_classes"])),
        "client_n_train": " ".join(map(str, cl["n_train"])),
        "lr": cfg.lr, "bytes_per_round": 0, "wall_s": round(time.perf_counter() - t_start, 2),
    }
    return summary, pd.DataFrame(rows_out)
