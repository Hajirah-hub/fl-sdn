"""Centralized-baseline diagnostics (seed 0): underfitting, leakage (random split), undertraining.

    python scripts/diag_central.py

A. normal chronological split, 30 epochs, lr 0.05 (the baseline)
B. random stratified 80/20 split of the same rows (train + test, y_multi >= 0), 30 epochs, lr 0.05
   -- leakage diagnostic only (paper Sec. III-C), never used for comparisons
C. normal split, 100 epochs, lr 0.1
Scaling and class weights come from each variant's own training rows.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from fedsdn.data.scaling import aggregate_feature_stats, client_feature_stats  # noqa: E402
from fedsdn.models import MLP  # noqa: E402
from fedsdn.sim import class_names, class_weights, load_flows  # noqa: E402

SEED = 0


def train_eval(X_raw, y, tr, te, C, epochs, lr, seed=SEED):
    torch.manual_seed(seed)
    mean, std = aggregate_feature_stats([client_feature_stats(X_raw[tr])])
    X = torch.from_numpy(((X_raw - mean) / std).astype(np.float32))
    yt = torch.from_numpy(y.astype(np.int64))
    cw = class_weights(np.bincount(y[tr], minlength=C))
    model = MLP(X.shape[1], C)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    gen = torch.Generator().manual_seed(seed)
    tr_t = torch.from_numpy(tr)
    for _ in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), generator=gen)]
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(model(X[idx]), yt[idx], weight=cw)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    out = {}
    with torch.no_grad():
        for name, rows in (("train", tr), ("test", te)):
            pred = model(X[rows]).argmax(1).numpy()
            out[f"{name}_f1"] = f1_score(y[rows], pred, labels=list(range(C)), average="macro", zero_division=0)
            out[f"{name}_recall"] = recall_score(y[rows], pred, labels=list(range(C)), average=None, zero_division=0)
    return out


def main():
    torch.set_num_threads(2)
    data = Path(os.environ.get("FEDSDN_DATA", "./data")) / "processed" / "insdn"
    meta = json.loads((data / "meta.json").read_text())
    info, X_raw, _ = load_flows(str(data))
    names = class_names(meta)
    C = len(names)
    y = info["y_multi"].to_numpy()
    ok = y >= 0
    tr = np.flatnonzero((info["split"] == "train").to_numpy() & ok)
    te = np.flatnonzero((info["split"] == "test").to_numpy() & ok)
    allrows = np.concatenate([tr, te])
    rtr, rte = train_test_split(allrows, test_size=0.2, stratify=y[allrows], random_state=SEED)

    variants = [
        ("A chrono 30ep lr0.05", tr, te, 30, 0.05),
        ("B random 30ep lr0.05", np.sort(rtr), np.sort(rte), 30, 0.05),
        ("C chrono 100ep lr0.1", tr, te, 100, 0.1),
    ]
    res = {}
    for name, a, b, ep, lr in variants:
        t0 = time.perf_counter()
        res[name] = train_eval(X_raw, y, a, b, C, ep, lr)
        print(f"{name}: train F1 {res[name]['train_f1']:.4f}  test F1 {res[name]['test_f1']:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    print("\n| variant | train F1 | test F1 | " + " | ".join(f"{n} train/test rec." for n in names) + " |")
    print("|---|---|---|" + "---|" * C)
    for name, r in res.items():
        cells = " | ".join(f"{a:.2f} / {b:.2f}" for a, b in zip(r["train_recall"], r["test_recall"]))
        print(f"| {name} | {r['train_f1']:.3f} | {r['test_f1']:.3f} | {cells} |")
    print("\ntest support:", dict(zip(names, np.bincount(y[te], minlength=C).tolist())),
          " random-split test support:", dict(zip(names, np.bincount(y[rte], minlength=C).tolist())))


if __name__ == "__main__":
    main()
