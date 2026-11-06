"""Build controller-level partitions for one dataset and print their heterogeneity statistics.

    python scripts/make_partitions.py --dataset insdn --K 8 --seeds 0-4
    python scripts/make_partitions.py --dataset insdn --settings P1_a0.1 P3_m1 --seeds 0
    python scripts/make_partitions.py --dataset insdn --report-only --show-clients P3_m1_K8_s0

Reads $FEDSDN_DATA/processed/<ds>/{flows.parquet, meta.json}; writes
$FEDSDN_DATA/partitions/<ds>/<scheme>_<param>_K<K>_s<seed>.{parquet,json}.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from fedsdn.data.partition import (  # noqa: E402
    SETTINGS, make_partition, partition_name, training_pool, write_partition,
)
from fedsdn.data.preprocess import SPECS, sha256_file  # noqa: E402


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def print_table(sidecars: list[dict]) -> None:
    by_tag: dict[str, list[dict]] = {}
    for s in sidecars:
        by_tag.setdefault(s["name"].rsplit("_K", 1)[0], []).append(s)
    for tag, rows in by_tag.items():
        print(f"\n{tag}")
        print(f"  {'seed':>4} {'JSD':>7} {'CV':>7} {'min':>6} {'max':>6} {'attempts':>8} {'moved':>6}")
        for s in sorted(rows, key=lambda r: r["seed"]):
            print(f"  {s['seed']:>4} {s['mean_pairwise_jsd']:>7.4f} {s['shard_size_cv']:>7.4f} "
                  f"{s['min_shard']:>6} {s['max_shard']:>6} {s['attempts']:>8} {s['rows_moved']:>6}")


def print_clients(s: dict) -> None:
    fams = s["families"]
    print(f"\n{s['name']}: per-client family counts (shard; validation rows in brackets)")
    print(f"  {'client':>6} " + " ".join(f"{f:>12}" for f in fams) + f" {'total':>7}")
    for k, row in s["client_family_counts"].items():
        val = s["client_val_family_counts"][k]
        cells = " ".join(f"{row[f]:>6} [{val[f]:>3}]" for f in fams)
        print(f"  {k:>6} {cells} {sum(row.values()):>7}")
    if "family_to_clients" in s:
        print(f"  m_eff = {s['m_eff']}; family -> clients: {s['family_to_clients']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="insdn", choices=sorted(SPECS))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--settings", nargs="*", help="setting tags, e.g. P0 P1_a0.1 (default: all)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--show-clients", nargs="*", default=[], help="partition names to detail")
    args = ap.parse_args()

    data = Path(os.environ.get("FEDSDN_DATA", "./data"))
    proc = data / "processed" / args.dataset
    out_dir = data / "partitions" / args.dataset
    settings = [s for s in SETTINGS if not args.settings or s.tag in args.settings]
    seeds = parse_seeds(args.seeds)

    sidecars = []
    if args.report_only:
        for st in settings:
            for seed in seeds:
                p = out_dir / f"{partition_name(st, args.K, seed)}.json"
                if p.exists():
                    sidecars.append(json.loads(p.read_text(encoding="utf-8")))
    else:
        meta = json.loads((proc / "meta.json").read_text(encoding="utf-8"))
        flows_path = proc / "flows.parquet"
        sha = sha256_file(flows_path)
        if sha != meta["flows_sha256"]:
            sys.exit(f"flows.parquet hash {sha[:12]} != meta.json {meta['flows_sha256'][:12]}")
        flows = pd.read_parquet(flows_path, columns=["row_id", "family", "split"])
        pool = training_pool(flows, meta["rare_class_decision"]["class_of_family"])
        benign = SPECS[args.dataset].benign_label
        for st in settings:
            for seed in seeds:
                part, side = make_partition(pool, st, args.K, seed, benign=benign,
                                            flows_sha256=sha, dataset=args.dataset)
                write_partition(out_dir, part, side)
                sidecars.append(side)
        print(f"wrote {len(sidecars)} partitions to {out_dir} ({len(pool)} training-pool rows)")

    print_table(sidecars)
    for name in args.show_clients:
        print_clients(json.loads((out_dir / f"{name}.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
