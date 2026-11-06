"""Tables and figures for the reduced study, both protocols side by side.

    python scripts/report_reduced.py          # -> results/report/tables.md, results/report/figures/*.png

Sources (under $FEDSDN_RESULTS):
  chrono  R = 30   chrono/{summary,rounds}.csv          (primary protocol; incl. poisoning)
  random  R = 100  r100/random/{summary,rounds}.csv     (secondary protocol headline)
  random  R = 30   random/{summary,rounds}.csv          (only for the R = 30 vs R = 100 comparison)
Headline = val-selected test macro-F1 (round with the best global validation macro-F1).
T95 = first round whose test macro-F1 >= 0.95 x the protocol's mean centralized val-selected
test macro-F1 (centralized trained for the same number of epochs as the protocol's rounds);
censored if never reached.
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STRATS = ["fedavg", "fedprox", "fednova", "fedadam", "krum", "median", "trimmed_mean"]
PARTS = ["P0", "P1_a0.5", "P3_m1", "P4_a0.5b0.5"]
PSHORT = {"P0": "P0", "P1_a0.5": "P1", "P3_m1": "P3", "P4_a0.5b0.5": "P4"}
PROTOS = ["chrono", "random"]
P4 = "P4_a0.5b0.5"


def pm(s, digits=3) -> str:
    s = pd.Series(s).dropna()
    if s.empty:
        return "–"
    sd = s.std(ddof=1) if len(s) > 1 else 0.0
    return f"{s.mean():.{digits}f} ± {sd:.{digits}f}"


def md(df: pd.DataFrame) -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in df.iterrows()]
    return "\n".join(out)


def add_t95(clean: pd.DataFrame, rounds: pd.DataFrame, thr: float) -> pd.DataFrame:
    clean = clean.copy()
    t95, cens = [], []
    for rid in clean.run_id:
        rr = rounds[rounds.run_id == rid].sort_values("round")
        hit = rr[rr.macro_f1 >= thr]
        t95.append(int(hit["round"].iloc[0]) if len(hit) else int(rr["round"].max()))
        cens.append(hit.empty)
    clean["t95"], clean["t95_censored"] = t95, np.array(cens, dtype=bool)
    return clean


def load(d: Path, label: str) -> dict:
    s = pd.read_csv(d / "summary.csv")
    r = pd.read_csv(d / "rounds.csv")
    central = s[s.kind == "central"]
    thr = 0.95 * central["sel_macro_f1"].mean()
    clean = add_t95(s[(s.kind == "fed") & (s.attack == "none")], r, thr)
    return {"label": label, "summary": s, "rounds": r, "central": central, "clean": clean, "thr": thr,
            "R": int(s["R"].max()), "poison": s[(s.kind == "fed") & (s.attack != "none")]}


def t95_cell(g) -> str:
    if g.empty:
        return "–"
    hit, n_c = g[~g.t95_censored], int(g.t95_censored.sum())
    if hit.empty:
        return f"cens. {n_c}/{len(g)}"
    mb = hit.t95.mean() * hit.bytes_per_round.mean() / 1e6
    return f"{hit.t95.mean():.1f} r / {mb:.1f} MB" + (f" (cens. {n_c}/{len(g)})" if n_c else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.environ.get("FEDSDN_RESULTS", "./results"))
    args = ap.parse_args()
    res = Path(args.results)
    rep = res / "report"
    figs = rep / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    r100 = res / "r100" / "random"
    data = {"chrono": load(res / "chrono", "chrono")}
    data["random"] = load(r100 if (r100 / "summary.csv").exists() else res / "random", "random")
    for p in PROTOS:
        data[p]["label"] = f"{p} R{data[p]['R']}"
        data[p]["clean"][["run_id", "strategy", "partition", "seed", "t95", "t95_censored"]].to_csv(
            rep / f"t95_{p}_R{data[p]['R']}.csv", index=False)

    def grid(fn, central_fn=None) -> pd.DataFrame:
        cols = {}
        for proto in PROTOS:
            clean = data[proto]["clean"]
            for p in PARTS:
                col = [fn(clean[(clean.strategy == s) & (clean.partition == p)]) for s in STRATS]
                if central_fn is not None:
                    col.append(central_fn(data[proto]["central"]) if p == "P0" else "")
                cols[f"{data[proto]['label']} {PSHORT[p]}"] = col
        idx = STRATS + (["centralized"] if central_fn is not None else [])
        return pd.DataFrame(cols, index=pd.Index(idx, name="strategy"))

    rc, rr_ = data["chrono"]["R"], data["random"]["R"]
    out = ["# Reduced study — InSDN multi-class, K = 8, E = 1, seeds 0–2", "",
           f"**chrono R{rc}** = primary chronological protocol, R = {rc} rounds (centralized: {rc} epochs). "
           f"**random R{rr_}** = secondary random stratified 80/20 split of the same rows, R = {rr_} rounds "
           f"(centralized: {rr_} epochs); a leakage diagnostic, not comparable with chrono. "
           f"Poisoning is chrono R{rc} only.", "",
           "Headline = **val-selected** test macro-F1: the round with the best global validation "
           "macro-F1 (pooled over all clients' validation sets); the test set is never used for "
           "selection. Mean ± std over 3 seeds. Centralized is one value per protocol (P0 column).", ""]

    out += ["## 1. Val-selected test macro-F1 (headline)", "",
            md(grid(lambda g: pm(g.sel_macro_f1), lambda c: pm(c.sel_macro_f1))), ""]

    def fin_auc(g):
        return f"{g.final_macro_f1.mean():.3f} / {g.auc_macro_f1.mean():.3f}" if len(g) else "–"

    out += ["## 2. Final-round / AUC test macro-F1 (means; AUC = mean over rounds 1..R)", "",
            md(grid(fin_auc, fin_auc)), ""]

    def recalls(g):
        return "/".join(f"{g[f'sel_recall_{c}'].mean():.2f}" for c in ("DoS", "Probe", "DDoS")) if len(g) else "–"

    out += ["## 3. Per-class test recall at the val-selected round, DoS / Probe / DDoS (mean over seeds)",
            "", md(grid(recalls, recalls)), ""]

    def bfa(g):
        return f"{g.sel_recall_BFA.mean():.2f}" if len(g) else "–"

    out += ["## 4. BFA test recall at the val-selected round (kept in macro-F1; listed separately)",
            "", md(grid(bfa, bfa)), ""]

    out += ["## 5. Worst-client validation macro-F1 at the val-selected round",
            "", "Per client: macro-F1 over the classes present in its validation set.", "",
            md(grid(lambda g: pm(g.sel_worst_client_val_f1))), ""]

    bpr = int(data["chrono"]["clean"].bytes_per_round.iloc[0])
    thr_txt = "; ".join(f"{data[p]['label']}: {data[p]['thr']:.4f} = 0.95 × "
                        f"{data[p]['central'].sel_macro_f1.mean():.4f} ({data[p]['R']}-epoch centralized)"
                        for p in PROTOS)
    out += ["## 6. T95 and bytes", "",
            f"Threshold: {thr_txt}. Bytes per round = params × 4 B × 2K = {bpr:,} ({bpr / 1e6:.2f} MB). "
            "Cell = mean T95 over seeds that reach it / MB to T95; cens. = seeds never reaching it.", "",
            md(grid(t95_cell)), ""]

    pz, cl = data["chrono"]["poison"], data["chrono"]["clean"]
    rows = {}
    for s in [x for x in STRATS if x in set(pz.strategy)]:
        base = cl[(cl.strategy == s) & (cl.partition == P4)]
        row = {"clean F1": pm(base.sel_macro_f1), "clean FNR": pm(base.sel_attack_fnr)}
        for a in ("sign_flip", "targeted"):
            g = pz[(pz.strategy == s) & (pz.attack == a)]
            row[f"{a} F1"] = pm(g.sel_macro_f1)
            row[f"{a} FNR"] = pm(g.sel_attack_fnr)
        rows[s] = row
    t = pd.DataFrame(rows).T
    t.index.name = "strategy"
    out += [f"## 7. Poisoning — chrono R{rc}, P4, f = 2 of 8 malicious (val-selected round)", "",
            "FNR = share of attack test flows predicted as Normal. Malicious clients' validation sets "
            "keep their true labels.", "", md(t), ""]

    r30_dir = res / "random"
    if rr_ != 30 and (r30_dir / "summary.csv").exists():
        d30 = load(r30_dir, "random R30")
        thr = data["random"]["thr"]
        c30 = add_t95(d30["clean"], d30["rounds"], thr)          # same threshold as R = 100
        c100 = data["random"]["clean"]
        rows = []
        for s in STRATS:
            for p in PARTS:
                a = c30[(c30.strategy == s) & (c30.partition == p)]
                b = c100[(c100.strategy == s) & (c100.partition == p)]
                rows.append({"strategy / partition": f"{s} {PSHORT[p]}",
                             "R=30 val-sel F1": pm(a.sel_macro_f1),
                             f"R={rr_} val-sel F1": pm(b.sel_macro_f1),
                             f"R={rr_} sel. round": (f"{b.sel_round.mean():.0f} ({b.sel_round.min()}–{b.sel_round.max()})"
                                                     if len(b) else "–"),
                             "R=30 T95": t95_cell(a), f"R={rr_} T95": t95_cell(b)})
        c_a, c_b = d30["central"], data["random"]["central"]
        rows.append({"strategy / partition": "centralized", "R=30 val-sel F1": pm(c_a.sel_macro_f1),
                     f"R={rr_} val-sel F1": pm(c_b.sel_macro_f1),
                     f"R={rr_} sel. round": f"{c_b.sel_round.mean():.0f} ({c_b.sel_round.min()}–{c_b.sel_round.max()})",
                     "R=30 T95": "", f"R={rr_} T95": ""})
        t = pd.DataFrame(rows).set_index("strategy / partition")
        out += [f"## 8. Random protocol, R = 30 vs R = {rr_}", "",
                f"Both T95 columns use the R = {rr_} threshold ({thr:.4f}). Centralized: 30 vs {rr_} epochs. "
                "Sel. round = val-selected round, mean (min–max).", "", md(t), ""]

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, proto in zip(axes, PROTOS):
        d = data[proto]
        for s in STRATS:
            ids = d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == P4)].run_id
            g = d["rounds"][d["rounds"].run_id.isin(ids)].groupby("round")["macro_f1"].agg(["mean", "std"])
            if g.empty:
                continue
            ax.plot(g.index, g["mean"], label=s)
            ax.fill_between(g.index, g["mean"] - g["std"].fillna(0), g["mean"] + g["std"].fillna(0), alpha=0.12)
        ax.axhline(d["central"].sel_macro_f1.mean(), color="k", ls="--", lw=1, label="centralized (val-sel.)")
        ax.axhline(d["thr"], color="gray", ls=":", lw=1, label="T95 threshold")
        ax.set(xlabel="round", ylabel="test macro-F1", title=f"{d['label']}: convergence under P4")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figs / "convergence_P4.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    w = 0.8 / len(STRATS)
    x = np.arange(len(PARTS))
    for ax, proto in zip(axes, PROTOS):
        d = data[proto]
        rare = d["summary"].rare_class.mode().iloc[0]
        for i, s in enumerate(STRATS):
            vals = [d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)].sel_rare_recall for p in PARTS]
            ax.bar(x + i * w - 0.4 + w / 2, [v.mean() for v in vals],
                   yerr=[v.std(ddof=1) if len(v) > 1 else 0 for v in vals], width=w, label=s, capsize=2)
        ax.axhline(d["central"].sel_rare_recall.mean(), color="k", ls="--", lw=1, label="centralized")
        ax.set_xticks(x, [PSHORT[p] for p in PARTS])
        ax.set(ylim=(0, 1.05), ylabel=f"recall of rarest training class ({rare})",
               title=f"{d['label']}: rare-class recall, val-selected round")
    axes[0].legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(figs / "rare_recall.png", dpi=150)
    plt.close(fig)

    out += ["Figures: report/figures/convergence_P4.png, report/figures/rare_recall.png", ""]
    text = "\n".join(out)
    (rep / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
