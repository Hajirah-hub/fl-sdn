"""Run the reduced study (plain PyTorch, no Flower). Skips runs already finished.

    python scripts/run_reduced.py                 # both protocols, 4 worker processes
    python scripts/run_reduced.py --dry           # 2 rounds, a few runs, results/dry/
    python scripts/run_reduced.py --suite r100    # random protocol, R = 100, all strategies -> results/r100/
    tail -f results/progress.log                  # live progress

Protocols: chrono (primary; centralized + 84 federated + 24 poisoning) and random (secondary;
centralized + 84 federated, no poisoning). Per run: $FEDSDN_RESULTS/<protocol>/runs/<run_id>.json
(summary; written last = done marker) and <run_id>_rounds.csv. After all runs:
$FEDSDN_RESULTS/<protocol>/summary.csv and rounds.csv.
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

import pandas as pd  # noqa: E402

from fedsdn.data.preprocess import atomic_write  # noqa: E402
from fedsdn.sim import PROTOCOL_SUFFIX, STRATEGIES, RunConfig, run  # noqa: E402

PARTITIONS = ("P0", "P1_a0.5", "P3_m1", "P4_a0.5b0.5")
SEEDS = (0, 1, 2)
POISON_STRATEGIES = ("fedavg", "krum", "median", "trimmed_mean")
ATTACKS = ("sign_flip", "targeted")


def protocol_configs(protocol: str, poisoning: bool) -> list[RunConfig]:
    cfgs = [RunConfig(kind="central", seed=s, protocol=protocol) for s in SEEDS]
    cfgs += [RunConfig(kind="fed", strategy=st, partition=p, seed=s, protocol=protocol)
             for st in STRATEGIES for p in PARTITIONS for s in SEEDS]
    if poisoning:
        cfgs += [RunConfig(kind="fed", strategy=st, partition="P4_a0.5b0.5", seed=s, attack=a, f=2,
                           protocol=protocol)
                 for st in POISON_STRATEGIES for a in ATTACKS for s in SEEDS]
    return cfgs


def all_configs() -> list[RunConfig]:
    return protocol_configs("chrono", True) + protocol_configs("random", False)


def r100_configs() -> list[RunConfig]:
    """Random protocol at R = 100: centralized (100 epochs) + all 7 strategies x 4 partitions x
    seeds 0-2. FedAvg/FedAdam were run first; finished runs are skipped."""
    cfgs = [RunConfig(kind="central", seed=s, protocol="random", R=100) for s in SEEDS]
    cfgs += [RunConfig(kind="fed", strategy=st, partition=p, seed=s, protocol="random", R=100)
             for st in ("fedavg", "fedadam") for p in PARTITIONS for s in SEEDS]
    cfgs += [RunConfig(kind="fed", strategy=st, partition=p, seed=s, protocol="random", R=100)
             for st in STRATEGIES if st not in ("fedavg", "fedadam") for p in PARTITIONS for s in SEEDS]
    return cfgs


def dry_configs() -> list[RunConfig]:
    cfgs = []
    for proto in ("chrono", "random"):
        cfgs.append(RunConfig(kind="central", seed=0, protocol=proto))
        cfgs += [RunConfig(kind="fed", strategy=st, partition="P4_a0.5b0.5", seed=0, protocol=proto)
                 for st in STRATEGIES]
    cfgs += [RunConfig(kind="fed", strategy="median", partition="P4_a0.5b0.5", seed=0, attack=a, f=2)
             for a in ATTACKS]
    return [replace(c, R=2) for c in cfgs]


_CTX = {}


def _init(data_dir, dataset, results_dir):
    import torch
    torch.set_num_threads(1)
    _CTX.update(data=Path(data_dir), dataset=dataset, results=Path(results_dir), meta={})


def _work(cfg: RunConfig):
    name = _CTX["dataset"] + PROTOCOL_SUFFIX[cfg.protocol]
    processed = _CTX["data"] / "processed" / name
    if name not in _CTX["meta"]:
        _CTX["meta"][name] = json.loads((processed / "meta.json").read_text())
    summary, rounds = run(cfg, str(processed), str(_CTX["data"] / "partitions" / name), _CTX["meta"][name])
    d = _CTX["results"] / cfg.protocol / "runs"
    atomic_write(d / f"{cfg.run_id}_rounds.csv", lambda t: rounds.to_csv(t, index=False))
    atomic_write(d / f"{cfg.run_id}.json", lambda t: Path(t).write_text(json.dumps(summary, indent=1)))
    return cfg, summary["sel_macro_f1"], summary["final_macro_f1"]


def collect(results: Path) -> dict[str, int]:
    n = {}
    for proto in PROTOCOL_SUFFIX:
        runs_dir = results / proto / "runs"
        js = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []
        if not js:
            continue
        summary = pd.DataFrame([json.loads(p.read_text()) for p in js])
        rounds = pd.concat([pd.read_csv(runs_dir / f"{p.stem}_rounds.csv") for p in js], ignore_index=True)
        atomic_write(results / proto / "summary.csv", lambda t: summary.to_csv(t, index=False))
        atomic_write(results / proto / "rounds.csv", lambda t: rounds.to_csv(t, index=False))
        n[proto] = len(js)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="insdn")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--suite", choices=["main", "r100"], default="main",
                    help="r100: random protocol, R = 100, centralized + all strategies -> results/r100/")
    args = ap.parse_args()

    data = Path(os.environ.get("FEDSDN_DATA", "./data"))
    results = Path(os.environ.get("FEDSDN_RESULTS", "./results"))
    if args.dry:
        results = results / "dry"
    elif args.suite == "r100":
        results = results / "r100"
    for proto in PROTOCOL_SUFFIX:
        (results / proto / "runs").mkdir(parents=True, exist_ok=True)

    cfgs = dry_configs() if args.dry else r100_configs() if args.suite == "r100" else all_configs()
    todo = [c for c in cfgs if not (results / c.protocol / "runs" / f"{c.run_id}.json").exists()]
    log = open(results / "progress.log", "a", buffering=1, encoding="utf-8")

    def emit(line: str) -> None:
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    emit(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} start: {len(cfgs)} runs, "
         f"{len(cfgs) - len(todo)} already done, {len(todo)} to go, {args.workers} workers -> {results}")
    sel_by = defaultdict(list)                 # (protocol, strategy) -> val-selected F1, clean runs
    t0 = time.perf_counter()
    with get_context("spawn").Pool(args.workers, _init, (str(data), args.dataset, str(results))) as pool:
        for i, (cfg, sel_f1, fin_f1) in enumerate(pool.imap_unordered(_work, todo), 1):
            el = time.perf_counter() - t0
            eta = el / i * (len(todo) - i)
            part = cfg.partition or "-"
            if cfg.attack:
                part += f"+{cfg.attack}"
            emit(f"[{i}/{len(todo)}] {el / 60:5.1f} min | ETA {eta / 60:5.1f} min | {cfg.protocol:<6} | "
                 f"{cfg.strategy:<12} | {part:<22} | s{cfg.seed} | val-sel F1 {sel_f1:.4f} | final F1 {fin_f1:.4f}")
            if not cfg.attack:
                sel_by[(cfg.protocol, cfg.strategy)].append(sel_f1)
            if i % 20 == 0 or i == len(todo):
                parts = []
                for proto in PROTOCOL_SUFFIX:
                    cells = [f"{st}={sum(v) / len(v):.3f}(n{len(v)})" for (p, st), v in sorted(sel_by.items())
                             if p == proto]
                    if cells:
                        parts.append(f"{proto}: " + " ".join(cells))
                emit(f"--- after {i} runs, mean val-selected F1 (clean runs): " + " ; ".join(parts))
    n = collect(results)
    emit(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} done: collected {n}")
    log.close()


if __name__ == "__main__":
    main()
