"""SECONDARY protocol: random stratified 80/20 split of the same rows as the chronological split.

    python scripts/make_random_split.py --dataset insdn

Reads  $FEDSDN_DATA/processed/<ds>/{flows.parquet, meta.json}
Writes $FEDSDN_DATA/processed/<ds>_random/{flows.parquet, meta.json}

Rows with split in {train, test} are re-split 80/20, stratified by partition family (rare
families -> Rare), random_state = --seed; capped rows stay "capped". Features, labels and the
cap are unchanged. Leakage diagnostic only: not comparable with the chronological test set.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fedsdn.data.preprocess import atomic_write, sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="insdn")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data = Path(os.environ.get("FEDSDN_DATA", "./data")) / "processed"
    src, dst = data / args.dataset, data / f"{args.dataset}_random"
    meta = json.loads((src / "meta.json").read_text())
    if sha256_file(src / "flows.parquet") != meta["flows_sha256"]:
        sys.exit("flows.parquet does not match meta.json")
    flows = pd.read_parquet(src / "flows.parquet")

    live = flows.index[flows["split"].isin(["train", "test"])]
    strat = flows.loc[live, "family"].map(meta["rare_class_decision"]["class_of_family"])
    tr, te = train_test_split(live, test_size=0.2, stratify=strat, random_state=args.seed)
    flows["split"] = flows["split"].astype("string")
    flows.loc[tr, "split"] = "train"
    flows.loc[te, "split"] = "test"

    atomic_write(dst / "flows.parquet", lambda t: flows.to_parquet(t, index=False))
    cls = flows["family"].map(meta["rare_class_decision"]["class_of_family"])
    meta_r = dict(meta)
    meta_r.update({
        "dataset": f"{args.dataset}_random",
        "protocol": "random_stratified_80_20",
        "protocol_note": "SECONDARY / leakage diagnostic: same rows as the chronological split, "
                         "re-split at random, stratified by partition family",
        "random_split_seed": args.seed,
        "source_flows_sha256": meta["flows_sha256"],
        "flows_sha256": sha256_file(dst / "flows.parquet"),
        "class_counts_per_split": {
            s: {str(k): int(v) for k, v in cls[flows["split"] == s].value_counts().sort_index().items()}
            for s in ("train", "test", "capped")
        },
    })
    for k in ("split_per_family", "split_method_per_family", "unsplit_families", "degenerate_grouping_flags"):
        meta_r.pop(k, None)
    atomic_write(dst / "meta.json", lambda t: Path(t).write_text(json.dumps(meta_r, indent=2)))
    print(f"wrote {dst}: train {len(tr)}, test {len(te)}")
    print(json.dumps(meta_r["class_counts_per_split"], indent=1))


if __name__ == "__main__":
    main()
