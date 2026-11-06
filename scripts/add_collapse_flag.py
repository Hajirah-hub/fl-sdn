"""Backfill the `collapsed` flag (fedsdn.sim.is_collapsed on the final-round per-class test
recalls) into finished run summaries of the v2 suites, then re-collect <suite>/summary.csv.
Runs finished after the flag was added already carry it; they are left unchanged.

    python scripts/add_collapse_flag.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedsdn.data.preprocess import atomic_write  # noqa: E402
from fedsdn.sim import is_collapsed  # noqa: E402
from run_study import collect  # noqa: E402

SUITES = ("tuning", "random", "chrono", "sweep", "poison_random", "lr_boundary", "lr_sensitivity")
CLASSES = ("Normal", "BFA", "DDoS", "DoS", "Probe")


def main() -> None:
    res = Path(os.environ.get("FEDSDN_RESULTS", "./results"))
    for suite in SUITES:
        d = res / suite / "runs"
        if not d.exists():
            continue
        added = flagged = 0
        for p in sorted(d.glob("*.json")):
            s = json.loads(p.read_text())
            if "collapsed" not in s:
                s["collapsed"] = is_collapsed([s[f"final_recall_{c}"] for c in CLASSES])
                atomic_write(p, lambda t, s=s: Path(t).write_text(json.dumps(s, indent=1)))
                added += 1
            flagged += bool(s["collapsed"])
        n = collect(res / suite)
        print(f"{suite:<15} runs {n:>4}  flag added {added:>4}  collapsed {flagged:>3}")


if __name__ == "__main__":
    main()
