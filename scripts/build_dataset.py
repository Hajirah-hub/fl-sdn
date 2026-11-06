"""Build data/processed/<ds>/{flows.parquet, meta.json} from data/raw/<ds>/ and print the report.

    python scripts/build_dataset.py --dataset insdn
    python scripts/build_dataset.py --dataset insdn --report-only     # print existing meta.json

Paths default to $FEDSDN_DATA (default ./data): raw from <data>/raw/<ds>, output to
<data>/processed/<ds>. Override with --raw-dir / --out-dir (e.g. Kaggle's input is read-only).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedsdn.data.preprocess import SPECS, Params, build, format_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="insdn", choices=sorted(SPECS))
    ap.add_argument("--raw-dir", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    data = Path(os.environ.get("FEDSDN_DATA", "./data"))
    raw_dir = args.raw_dir or data / "raw" / args.dataset
    out_dir = args.out_dir or data / "processed" / args.dataset

    if args.report_only:
        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
    else:
        t0 = time.perf_counter()
        meta = build(raw_dir, out_dir, SPECS[args.dataset], Params())
        print(f"wrote {out_dir / 'flows.parquet'} and meta.json in {time.perf_counter() - t0:.1f} s")
    print(format_report(meta))


if __name__ == "__main__":
    main()
