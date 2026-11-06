"""Upgraded study (v2): tuning, SCAFFOLD, 5 seeds. Plain PyTorch (fedsdn.sim), resumable.

    python scripts/run_study.py --suite tuning    # 8 strategies + centralized x 12 trials, seed 99
    python scripts/run_study.py --suite random    # random protocol, R = 100, 8 x 4 partitions x seeds 0-4
    python scripts/run_study.py --suite chrono    # chronological, R = 30, same grid + poisoning (P4)
    python scripts/run_study.py --suite sweep     # random, R = 100: FedAvg/FedAdam on P1 a0.1/1/10, P2 b0.5
    python scripts/run_study.py --suite local     # random, local-only baseline, 100 epochs per client
    python scripts/run_study.py --suite poison_random   # random P3 m1, R = 100, 4 rules x 2 attacks
    python scripts/run_study.py --suite lr_boundary     # tuning seed: lr 0.4 / 0.6 vs frozen best (record only)
    python scripts/run_study.py --suite lr_sensitivity  # random, R = 100: FedAvg / FedProx at client lr 0.6
    python scripts/run_study.py --suite tuning --dry    # 2 rounds, 1 trial -> results/dry_v2/
    tail -f results/progress.log

Per run: $FEDSDN_RESULTS/<suite>/runs/<run_id>.json (summary; written last = done marker) and
<run_id>_rounds.csv; after a suite: <suite>/summary.csv and rounds.csv. Every suite except
`tuning` reads the frozen values in configs/tuning/<dataset>.yaml.

Tuning (paper Section III-H): 12 random-search trials per strategy, log-uniform. The 12 client
learning rates in [1e-3, 3e-1] are drawn once (rng 99) and shared by all strategies (trial i uses
lr_i everywhere); FedProx mu in [1e-3, 1] (rng [99, 1]), FedAdam server lr in [1e-2, 1] (rng
[99, 2]) and eps in [1e-3, 1e-1] (rng [99, 3]) are drawn per trial. R = 30, random protocol,
seed 99 (own partitions), P1 a0.5 / P3 m1 / P4. Score = mean over the 3 partitions of the best
pooled validation macro-F1 over rounds (the test set is never read for tuning). A trial in which
any run diverges (non-finite weights) is not eligible. Ties -> earliest trial. The centralized
baseline is tuned the same way over the same 12 learning rates (one run per trial, its own 10%
validation split).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from fedsdn.data.preprocess import atomic_write  # noqa: E402
from fedsdn.sim import PROTOCOL_SUFFIX, STRATEGIES, RunConfig, run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAIN_PARTITIONS = ("P0", "P1_a0.5", "P3_m1", "P4_a0.5b0.5")
TUNE_PARTITIONS = ("P1_a0.5", "P3_m1", "P4_a0.5b0.5")
SWEEP_PARTITIONS = ("P1_a0.1", "P1_a1", "P1_a10", "P2_b0.5")
SEEDS = (0, 1, 2, 3, 4)
TUNE_SEED, N_TRIALS = 99, 12
POISON_STRATEGIES = ("fedavg", "krum", "median", "trimmed_mean")
BOUNDARY_STRATEGIES = ("fedavg", "fednova", "fedprox", "scaffold")
BOUNDARY_LRS = (0.4, 0.6)
SENSITIVITY_STRATEGIES, SENSITIVITY_LR = ("fedavg", "fedprox"), 0.6
ATTACKS = ("sign_flip", "targeted")
TUNED_KEYS = {"lr", "mu", "server_lr", "eps"}
TUNE_LIKE = ("tuning", "lr_boundary")          # suites scored on validation only


# --------------------------------------------------------------------------------------------
# Tuning
# --------------------------------------------------------------------------------------------


def log_uniform(rng_seed, lo: float, hi: float, n: int = N_TRIALS) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def trial_values() -> list[dict]:
    lr = log_uniform(TUNE_SEED, 1e-3, 3e-1)
    mu = log_uniform([TUNE_SEED, 1], 1e-3, 1.0)
    slr = log_uniform([TUNE_SEED, 2], 1e-2, 1.0)
    eps = log_uniform([TUNE_SEED, 3], 1e-3, 1e-1)
    return [{"lr": float(lr[i]), "mu": float(mu[i]), "server_lr": float(slr[i]), "eps": float(eps[i])}
            for i in range(N_TRIALS)]


def trial_params(strategy: str, t: dict) -> dict:
    """Hyperparameters actually searched for a strategy (others keep their defaults)."""
    if strategy == "fedprox":
        return {"lr": t["lr"], "mu": t["mu"]}
    if strategy == "fedadam":
        return {"lr": t["lr"], "server_lr": t["server_lr"], "eps": t["eps"]}
    return {"lr": t["lr"]}


def tuning_configs() -> list[RunConfig]:
    cfgs = []
    for i, t in enumerate(trial_values()):
        tag = f"t{i:02d}"
        cfgs.append(RunConfig(kind="central", seed=TUNE_SEED, protocol="random", R=30, tag=tag,
                              **trial_params("central", t)))
        cfgs += [RunConfig(kind="fed", strategy=st, partition=p, seed=TUNE_SEED, protocol="random",
                           R=30, tag=tag, **trial_params(st, t))
                 for st in STRATEGIES for p in TUNE_PARTITIONS]
    return cfgs


def freeze_tuning(results: Path, dataset: str) -> None:
    """Score the trials, write <suite>/trials.csv + tuning_summary.csv and configs/tuning/<ds>.yaml."""
    s = pd.read_csv(results / "summary.csv")
    s["trial"] = s["tag"].str[1:].astype(int)
    s["score_run"] = s["sel_val_macro_f1"]
    g = s.groupby(["strategy", "trial"])
    trials = g.agg(score=("score_run", "mean"), n_runs=("score_run", "size"),
                   diverged=("diverged", "any"), lr=("lr", "first"), mu=("mu", "first"),
                   server_lr=("server_lr", "first"), eps=("eps", "first"),
                   test_f1_not_used=("sel_macro_f1", "mean")).reset_index()
    for st in trials["strategy"].unique():
        if st != "fedprox":
            trials.loc[trials["strategy"] == st, "mu"] = np.nan
        if st != "fedadam":
            trials.loc[trials["strategy"] == st, ["server_lr", "eps"]] = np.nan
    trials["eligible"] = ~trials["diverged"]
    trials["rank"] = (trials.assign(sc=trials["score"].where(trials["eligible"], -np.inf))
                      .sort_values(["strategy", "sc", "trial"], ascending=[True, False, True])
                      .groupby("strategy").cumcount() + 1)
    atomic_write(results / "trials.csv", lambda t: trials.sort_values(["strategy", "trial"])
                 .to_csv(t, index=False))

    chosen, rows = {}, []
    for st, d in trials.groupby("strategy"):
        best = d[d["rank"] == 1].iloc[0]
        ok = d[d["eligible"]]["score"]
        vals = {k: float(best[k]) for k in trial_params(st, {k: 0 for k in TUNED_KEYS})}
        chosen[st] = {**{k: float(f"{v:.6g}") for k, v in vals.items()},
                      "trial": int(best["trial"]), "val_score": round(float(best["score"]), 4)}
        rows.append({"strategy": st, **vals, "trial": int(best["trial"]), "best": best["score"],
                     "median": ok.median(), "min": ok.min(), "max": ok.max(),
                     "q25": ok.quantile(0.25), "q75": ok.quantile(0.75),
                     "second_best": ok.sort_values(ascending=False).iloc[1] if len(ok) > 1 else np.nan,
                     "n_diverged": int(d["diverged"].sum()), "n_trials": len(d)})
    atomic_write(results / "tuning_summary.csv", lambda t: pd.DataFrame(rows).to_csv(t, index=False))

    doc = {
        "dataset": dataset, "protocol": "random", "task": "multi", "seed": TUNE_SEED, "R": 30,
        "partitions": list(TUNE_PARTITIONS), "n_trials": N_TRIALS,
        "search": {"lr": "log-uniform [1e-3, 3e-1], 12 values drawn once (rng 99), shared",
                   "mu": "FedProx, log-uniform [1e-3, 1] (rng [99, 1])",
                   "server_lr": "FedAdam, log-uniform [1e-2, 1] (rng [99, 2])",
                   "eps": "FedAdam, log-uniform [1e-3, 1e-1] (rng [99, 3])"},
        "score": "mean over partitions of the best pooled validation macro-F1 over rounds; "
                 "diverged trials ineligible; ties -> earliest trial",
        "fixed": {"E": 1, "batch": 256, "K": 8, "f_hat": 2, "beta1": 0.9, "beta2": 0.99,
                  "scaffold_global_lr": 1.0},
        "strategies": chosen,
    }
    out = ROOT / "configs" / "tuning" / f"{dataset}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, lambda t: Path(t).write_text(
        "# Frozen by scripts/run_study.py --suite tuning. Do not edit by hand.\n"
        + yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"))


def load_tuned(dataset: str) -> dict[str, dict]:
    p = ROOT / "configs" / "tuning" / f"{dataset}.yaml"
    if not p.exists():
        sys.exit(f"{p} missing: run --suite tuning first")
    return {st: {k: v for k, v in d.items() if k in TUNED_KEYS}
            for st, d in yaml.safe_load(p.read_text(encoding="utf-8"))["strategies"].items()}


# --------------------------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------------------------


def suite_configs(suite: str, tuned: dict[str, dict]) -> list[RunConfig]:
    def fed(st, **kw):
        return RunConfig(kind="fed", strategy=st, **tuned[st], **kw)

    def central(**kw):
        return RunConfig(kind="central", **tuned["central"], **kw)

    if suite == "random":
        cfgs = [central(seed=s, protocol="random", R=100) for s in SEEDS]
        return cfgs + [fed(st, partition=p, seed=s, protocol="random", R=100)
                       for st in STRATEGIES for p in MAIN_PARTITIONS for s in SEEDS]
    if suite == "chrono":
        cfgs = [central(seed=s, protocol="chrono", R=30) for s in SEEDS]
        cfgs += [fed(st, partition=p, seed=s, protocol="chrono", R=30)
                 for st in STRATEGIES for p in MAIN_PARTITIONS for s in SEEDS]
        return cfgs + [fed(st, partition="P4_a0.5b0.5", seed=s, protocol="chrono", R=30, attack=a, f=2)
                       for st in POISON_STRATEGIES for a in ATTACKS for s in SEEDS]
    if suite == "sweep":
        return [fed(st, partition=p, seed=s, protocol="random", R=100)
                for st in ("fedavg", "fedadam") for p in SWEEP_PARTITIONS for s in SEEDS]
    if suite == "local":
        return [RunConfig(kind="local", strategy="local", partition=p, seed=s, protocol="random", R=100,
                          **tuned["central"]) for p in MAIN_PARTITIONS for s in SEEDS]
    if suite == "poison_random":
        return [fed(st, partition="P3_m1", seed=s, protocol="random", R=100, attack=a, f=2)
                for st in POISON_STRATEGIES for a in ATTACKS for s in SEEDS]
    if suite == "lr_sensitivity":     # sensitivity only, not the headline: frozen values except lr
        return [RunConfig(kind="fed", strategy=st, partition=p, seed=s, protocol="random", R=100,
                          tag=f"lr{SENSITIVITY_LR}", **{**tuned[st], "lr": SENSITIVITY_LR})
                for st in SENSITIVITY_STRATEGIES for p in MAIN_PARTITIONS for s in SEEDS]
    if suite == "lr_boundary":        # same setup as tuning; only the client lr changes
        return [RunConfig(kind="fed", strategy=st, partition=p, seed=TUNE_SEED, protocol="random", R=30,
                          tag=f"lr{lr}", **{**tuned[st], "lr": lr})
                for st in BOUNDARY_STRATEGIES for lr in BOUNDARY_LRS for p in TUNE_PARTITIONS]
    raise ValueError(suite)


def lr_boundary_report(results: Path) -> pd.DataFrame:
    """Compare lr 0.4 / 0.6 with the frozen best trial (same score definition; frozen values untouched)."""
    s = pd.read_csv(results / "summary.csv")
    trials = pd.read_csv(results.parent / "tuning" / "trials.csv")
    frozen = trials[trials["rank"] == 1].set_index("strategy")
    g = s.groupby(["strategy", "lr"]).agg(score=("sel_val_macro_f1", "mean"), n_runs=("run_id", "size"),
                                          diverged=("diverged", "any")).reset_index()
    g["frozen_lr"] = g.strategy.map(frozen["lr"])
    g["frozen_score"] = g.strategy.map(frozen["score"])
    g["diff_vs_frozen"] = g.score - g.frozen_score
    g["beats_frozen"] = ~g.diverged & (g.score > g.frozen_score)
    atomic_write(results / "lr_boundary.csv", lambda t: g.to_csv(t, index=False))
    return g


# --------------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------------

_CTX = {}


def _init(data_dir, dataset, results_dir):
    import torch
    torch.set_num_threads(1)
    _CTX.update(data=Path(data_dir), dataset=dataset, results=Path(results_dir), meta={})


def captured_families(sidecar_path: Path, malicious: list[int]) -> tuple[list[str], list[str]]:
    """Attack families (Normal and the masked Rare group excluded) all of whose holders are
    malicious, and the subset with a single (malicious) holder."""
    side = json.loads(sidecar_path.read_text(encoding="utf-8"))
    f2c = side.get("family_to_clients", {})
    mal = set(malicious)
    cap = [f for f, hs in f2c.items() if f not in ("Normal", "Rare") and set(hs) <= mal]
    return cap, [f for f in cap if len(f2c[f]) == 1]


def _work(cfg: RunConfig):
    name = _CTX["dataset"] + PROTOCOL_SUFFIX[cfg.protocol]
    processed = _CTX["data"] / "processed" / name
    parts = _CTX["data"] / "partitions" / name
    if name not in _CTX["meta"]:
        _CTX["meta"][name] = json.loads((processed / "meta.json").read_text())
    summary, rounds = run(cfg, str(processed), str(parts), _CTX["meta"][name])
    if cfg.attack:
        mal = [int(m) for m in summary["malicious"].split()]
        cap, sole = captured_families(parts / f"{cfg.partition}_K{cfg.K}_s{cfg.seed}.json", mal)
        summary.update(captured_families=" ".join(cap), sole_holder_families=" ".join(sole),
                       malicious_sole_holder=bool(sole))
    d = _CTX["results"] / "runs"
    atomic_write(d / f"{cfg.run_id}_rounds.csv", lambda t: rounds.to_csv(t, index=False))
    atomic_write(d / f"{cfg.run_id}.json", lambda t: Path(t).write_text(json.dumps(summary, indent=1)))
    return cfg, summary["sel_macro_f1"], summary["final_macro_f1"], summary.get("sel_val_macro_f1")


def collect(results: Path) -> int:
    runs_dir = results / "runs"
    js = sorted(runs_dir.glob("*.json"))
    if not js:
        return 0
    summary = pd.DataFrame([json.loads(p.read_text()) for p in js])
    rounds = pd.concat([pd.read_csv(runs_dir / f"{p.stem}_rounds.csv") for p in js], ignore_index=True)
    atomic_write(results / "summary.csv", lambda t: summary.to_csv(t, index=False))
    atomic_write(results / "rounds.csv", lambda t: rounds.to_csv(t, index=False))
    return len(js)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True,
                    choices=["tuning", "random", "chrono", "sweep", "local", "poison_random",
                             "lr_boundary", "lr_sensitivity"])
    ap.add_argument("--dataset", default="insdn")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true", help="2 rounds, a few runs, results/dry_v2/")
    args = ap.parse_args()

    data = Path(os.environ.get("FEDSDN_DATA", "./data"))
    base = Path(os.environ.get("FEDSDN_RESULTS", "./results"))
    if args.dry:
        base = base / "dry_v2"
    results = base / args.suite
    (results / "runs").mkdir(parents=True, exist_ok=True)

    if args.suite == "tuning":
        cfgs = tuning_configs()
    else:
        tuned = load_tuned(args.dataset) if not args.dry else defaultdict(dict)
        cfgs = suite_configs(args.suite, tuned)
    if args.dry:
        seen, keep = set(), []
        for c in cfgs:                                    # one run per (kind, strategy, attack)
            key = (c.kind, c.strategy, c.attack)
            if key not in seen:
                seen.add(key)
                keep.append(replace(c, R=2))
        cfgs = keep
    todo = [c for c in cfgs if not (results / "runs" / f"{c.run_id}.json").exists()]
    log = open(base / "progress.log", "a", buffering=1, encoding="utf-8")

    def emit(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} [{args.suite}] start: {len(cfgs)} runs, "
         f"{len(cfgs) - len(todo)} already done, {len(todo)} to go, {args.workers} workers -> {results}")
    sel_by = defaultdict(list)
    t0 = time.perf_counter()
    with get_context("spawn").Pool(args.workers, _init, (str(data), args.dataset, str(results))) as pool:
        for i, (cfg, sel_f1, fin_f1, val_f1) in enumerate(pool.imap_unordered(_work, todo), 1):
            el = time.perf_counter() - t0
            eta = el / i * (len(todo) - i)
            part = cfg.partition or "-"
            if cfg.attack:
                part += f"+{cfg.attack}"
            extra = f" | {cfg.tag} lr {cfg.lr:.4g} | best val F1 {val_f1:.4f}" if args.suite in TUNE_LIKE \
                else f" | val-sel F1 {sel_f1:.4f} | final F1 {fin_f1:.4f}"
            emit(f"[{args.suite} {i}/{len(todo)}] {el / 60:5.1f} min | ETA {eta / 60:5.1f} min | "
                 f"{cfg.protocol:<6} | {cfg.strategy:<12} | {part:<22} | s{cfg.seed}{extra}")
            key = cfg.strategy + (f"+{cfg.attack}" if cfg.attack else "")
            sel_by[key].append(val_f1 if args.suite in TUNE_LIKE else sel_f1)
            if i % 20 == 0 or i == len(todo):
                what = "best val F1" if args.suite in TUNE_LIKE else "val-selected test F1"
                cells = [f"{k}={np.mean(v):.3f}(n{len(v)})" for k, v in sorted(sel_by.items())]
                emit(f"--- [{args.suite}] after {i} runs, mean {what}: " + " ".join(cells))
    n = collect(results)
    if n and (results / "summary.csv").exists():
        sm = pd.read_csv(results / "summary.csv")
        bad = {c: sm[c].fillna(False).astype(bool) for c in ("diverged", "collapsed") if c in sm}
        if bad:
            emit(f"--- [{args.suite}] " + ", ".join(f"{c}: {int(v.sum())}/{len(sm)}" for c, v in bad.items())
                 + (" -> " + " ".join(sm.loc[bad.get("collapsed", bad["diverged"]), "run_id"]) if
                    bad.get("collapsed", bad["diverged"]).any() else ""))
    if args.suite == "tuning" and not args.dry:
        freeze_tuning(results, args.dataset)
        emit(f"--- [tuning] frozen values -> configs/tuning/{args.dataset}.yaml")
    if args.suite == "lr_boundary" and not args.dry:
        for r in lr_boundary_report(results).itertuples():
            emit(f"--- [lr_boundary] {r.strategy:<9} lr {r.lr:<4} score {r.score:.4f} vs frozen "
                 f"{r.frozen_score:.4f} (lr {r.frozen_lr:.4g}) diff {r.diff_vs_frozen:+.4f} "
                 f"diverged {r.diverged} beats {r.beats_frozen}")
    emit(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} [{args.suite}] done: collected {n} runs")
    log.close()


if __name__ == "__main__":
    main()
