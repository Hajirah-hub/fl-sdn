"""Paper statistics, figures and tables from the FINAL results (never results/dry).

    python scripts/paper_outputs.py

Inputs ($FEDSDN_RESULTS): chrono/ (R = 30, incl. poisoning), r100/random/ (random protocol, R = 100),
random/ (random protocol, R = 30: R = 30 vs 100 comparison and the leakage diagnostic), and the
partition sidecars in $FEDSDN_DATA/partitions/. Outputs: results/paper_tables/*.csv,
results/paper_figures/*.png (300 dpi, IEEE single column 3.5 in unless noted, fonts >= 8 pt,
Okabe-Ito colorblind-safe palette, fixed color per strategy, no titles).

Leakage diagnostic (fig1, table_leakage_diagnostic): centralized baseline, 30 epochs on both
protocols (same training budget), seeds 0-2, val-selected epoch.
Statistics: Friedman test over the 7 strategies on val-selected test macro-F1 of the random
protocol at R = 100, blocks = partition x seed (12); Nemenyi post-hoc; alpha = 0.05. Planned
pairwise tests: Wilcoxon signed-rank of each strategy vs FedAvg over the same 12 blocks,
two-sided, Holm-corrected over 6 comparisons; effect sizes = median difference and matched-pairs
rank-biserial correlation (blocks sharing a partition are not fully independent).
Rounding: 3 decimals; p-values keep 3 significant digits. Error bars: mean +/- std over seeds,
clipped to [0, 1].
"""

import json
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scikit_posthocs as sp  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_reduced import add_t95  # noqa: E402

STRATS = ["fedavg", "fedprox", "fednova", "fedadam", "krum", "median", "trimmed_mean"]
LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova", "fedadam": "FedAdam",
         "krum": "Krum", "median": "Median", "trimmed_mean": "Trimmed mean"}
# Okabe-Ito palette, one fixed color per strategy; centralized = black
COLOR = {"fedavg": "#0072B2", "fedprox": "#56B4E9", "fednova": "#009E73", "fedadam": "#D55E00",
         "krum": "#CC79A7", "median": "#E69F00", "trimmed_mean": "#F0E442"}
LSTYLE = {"fedavg": "-", "fedprox": "--", "fednova": "-", "fedadam": "-", "krum": "-.",
          "median": "-", "trimmed_mean": ":"}
PARTS = ["P0", "P1_a0.5", "P3_m1", "P4_a0.5b0.5"]
PLABEL = {"P0": "P0 (IID)", "P1_a0.5": "P1 (α=0.5)", "P3_m1": "P3 (m=1)", "P4_a0.5b0.5": "P4 (α=β=0.5)"}
CLASSES = ["Normal", "BFA", "DDoS", "DoS", "Probe"]
P4 = "P4_a0.5b0.5"
SINGLE, DOUBLE = 3.5, 7.16
Q05_K7 = 2.949            # Nemenyi critical value q_0.05 for k = 7 (Demsar 2006, Table 5)


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.dpi": 300, "figure.dpi": 100,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def r3(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(3)


def load(d: Path) -> dict:
    s = pd.read_csv(d / "summary.csv")
    r = pd.read_csv(d / "rounds.csv")
    central = s[s.kind == "central"]
    thr = 0.95 * central["sel_macro_f1"].mean()
    clean = add_t95(s[(s.kind == "fed") & (s.attack == "none")], r, thr)
    return {"summary": s, "rounds": r, "central": central, "clean": clean, "thr": thr,
            "R": int(s["R"].max()), "poison": s[(s.kind == "fed") & (s.attack != "none")]}


def ms(x: pd.Series) -> tuple[float, float]:
    x = x.dropna()
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def clip_err(means, sds) -> np.ndarray:
    """Asymmetric error bars: mean +/- std, clipped to the [0, 1] score range."""
    m, sd = np.asarray(means, float), np.nan_to_num(np.asarray(sds, float))
    return np.vstack([np.minimum(sd, m), np.minimum(sd, 1 - m)])


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    res = Path(os.environ.get("FEDSDN_RESULTS", "./results"))
    data_dir = Path(os.environ.get("FEDSDN_DATA", "./data"))
    tab, figd = res / "paper_tables", res / "paper_figures"
    tab.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    style()

    chrono = load(res / "chrono")
    rnd = load(res / "r100" / "random")
    rnd30 = load(res / "random")
    assert chrono["R"] == 30 and rnd["R"] == 100 and rnd30["R"] == 30
    protos = {"chronological": chrono, "random": rnd}

    # ---------------------------------------------------------------- C. tables
    rows = []
    for pname, d in protos.items():
        for s in STRATS:
            for p in PARTS:
                g = d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)]
                m, sd = ms(g.sel_macro_f1)
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "macro_f1_mean": m, "macro_f1_std": sd, "n_seeds": len(g)})
        m, sd = ms(d["central"].sel_macro_f1)
        rows.append({"protocol": pname, "rounds": d["R"], "strategy": "centralized", "partition": "all",
                     "macro_f1_mean": m, "macro_f1_std": sd, "n_seeds": len(d["central"])})
    t_head = pd.DataFrame(rows)
    r3(t_head).to_csv(tab / "table_valselected_macro_f1.csv", index=False)

    rows = []
    for pname, d in protos.items():
        for s in STRATS + ["centralized"]:
            for p in (PARTS if s != "centralized" else ["all"]):
                g = d["central"] if s == "centralized" else d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)]
                fm, fs = ms(g.final_macro_f1)
                am, as_ = ms(g.auc_macro_f1)
                sm, ss = ms(g.sel_round)
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "final_macro_f1_mean": fm, "final_macro_f1_std": fs,
                             "auc_macro_f1_mean": am, "auc_macro_f1_std": as_,
                             "selected_round_mean": sm, "selected_round_std": ss})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_final_auc_macro_f1.csv", index=False)

    rows = []
    for pname, d in protos.items():
        for s in STRATS + ["centralized"]:
            for p in (PARTS if s != "centralized" else ["all"]):
                g = d["central"] if s == "centralized" else d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)]
                for c in CLASSES:
                    m, sd = ms(g[f"sel_recall_{c}"])
                    rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                                 "class": c, "recall_mean": m, "recall_std": sd})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_per_class_recall.csv", index=False)

    rows = []
    for pname, d in protos.items():
        for s in STRATS:
            for p in PARTS:
                g = d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)]
                m, sd = ms(g.sel_worst_client_val_f1)
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "worst_client_val_macro_f1_mean": m, "worst_client_val_macro_f1_std": sd})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_worst_client_val_f1.csv", index=False)

    rows = []
    for pname, d in protos.items():
        for s in STRATS:
            for p in PARTS:
                g = d["clean"][(d["clean"].strategy == s) & (d["clean"].partition == p)]
                hit = g[~g.t95_censored]
                bpr = int(g.bytes_per_round.iloc[0])
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "t95_threshold": d["thr"], "n_reached": len(hit), "n_censored": int(g.t95_censored.sum()),
                             "t95_rounds_mean_reached": hit.t95.mean() if len(hit) else np.nan,
                             "t95_rounds_std_reached": hit.t95.std(ddof=1) if len(hit) > 1 else (0.0 if len(hit) else np.nan),
                             "mb_to_t95_mean_reached": hit.t95.mean() * bpr / 1e6 if len(hit) else np.nan,
                             "bytes_per_round": bpr, "mb_per_round": bpr / 1e6})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_t95_bytes.csv", index=False)

    rows = []
    for s in ["fedavg", "krum", "median", "trimmed_mean"]:
        conds = {"clean": chrono["clean"][(chrono["clean"].strategy == s) & (chrono["clean"].partition == P4)]}
        for a in ("sign_flip", "targeted"):
            conds[a] = chrono["poison"][(chrono["poison"].strategy == s) & (chrono["poison"].attack == a)]
        for a, g in conds.items():
            fm, fs = ms(g.sel_macro_f1)
            nm, ns = ms(g.sel_attack_fnr)
            rows.append({"protocol": "chronological", "rounds": 30, "partition": P4, "strategy": s,
                         "attack": a, "malicious_clients": 0 if a == "clean" else 2,
                         "macro_f1_mean": fm, "macro_f1_std": fs, "attack_fnr_mean": nm, "attack_fnr_std": ns})
    t_poison = pd.DataFrame(rows)
    r3(t_poison).to_csv(tab / "table_poisoning.csv", index=False)

    c30 = add_t95(rnd30["clean"], rnd30["rounds"], rnd["thr"])
    rows = []
    for s in STRATS:
        for p in PARTS:
            a = c30[(c30.strategy == s) & (c30.partition == p)]
            b = rnd["clean"][(rnd["clean"].strategy == s) & (rnd["clean"].partition == p)]
            am, as_ = ms(a.sel_macro_f1)
            bm, bs = ms(b.sel_macro_f1)
            rows.append({"protocol": "random", "strategy": s, "partition": p,
                         "r30_macro_f1_mean": am, "r30_macro_f1_std": as_,
                         "r100_macro_f1_mean": bm, "r100_macro_f1_std": bs,
                         "r100_selected_round_mean": b.sel_round.mean(),
                         "r100_selected_round_min": b.sel_round.min(), "r100_selected_round_max": b.sel_round.max(),
                         "t95_threshold": rnd["thr"],
                         "r30_t95_n_reached": int((~a.t95_censored).sum()),
                         "r100_t95_n_reached": int((~b.t95_censored).sum()),
                         "r100_t95_rounds_mean_reached": b[~b.t95_censored].t95.mean() if (~b.t95_censored).any() else np.nan})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_random_r30_vs_r100.csv", index=False)

    rows = []
    for pname, d in (("chronological", chrono), ("random", rnd30)):
        g = d["central"]
        row = {"protocol": pname, "epochs": 30, "n_seeds": len(g)}
        row["macro_f1_mean"], row["macro_f1_std"] = ms(g.sel_macro_f1)
        for c in CLASSES:
            row[f"recall_{c}_mean"], row[f"recall_{c}_std"] = ms(g[f"sel_recall_{c}"])
        rows.append(row)
    t_leak = pd.DataFrame(rows)
    r3(t_leak).to_csv(tab / "table_leakage_diagnostic.csv", index=False)

    rows = []
    for pdir, pname in (("insdn", "chronological"), ("insdn_random", "random")):
        for p in PARTS:
            for seed in (0, 1, 2):
                sc = json.loads((data_dir / "partitions" / pdir / f"{p}_K8_s{seed}.json").read_text())
                rows.append({"protocol": pname, "partition": p, "seed": seed, "K": sc["K"],
                             "mean_pairwise_jsd": sc["mean_pairwise_jsd"], "shard_size_cv": sc["shard_size_cv"],
                             "min_shard": sc["min_shard"], "max_shard": sc["max_shard"],
                             "attempts": sc["attempts"], "rows_moved": sc["rows_moved"]})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_partition_heterogeneity.csv", index=False)

    # ---------------------------------------------------------------- A. statistics
    cl = rnd["clean"]
    blocks = cl.pivot_table(index=["partition", "seed"], columns="strategy", values="sel_macro_f1")[STRATS]
    assert blocks.shape == (12, 7) and not blocks.isna().any().any()
    stat, pval = friedmanchisquare(*[blocks[s].to_numpy() for s in STRATS])
    ranks = blocks.rank(axis=1, ascending=False)          # rank 1 = highest macro-F1
    mean_ranks = ranks.mean().sort_values()
    k, n = len(STRATS), len(blocks)
    cd = Q05_K7 * np.sqrt(k * (k + 1) / (6 * n))
    nem = sp.posthoc_nemenyi_friedman(blocks.to_numpy())
    nem.index = nem.columns = STRATS
    pd.DataFrame([{"test": "Friedman", "protocol": "random", "rounds": 100, "metric": "val-selected test macro-F1",
                   "k_strategies": k, "n_blocks": n, "blocks": "partition x seed",
                   "chi2_statistic": round(stat, 3), "p_value": float(f"{pval:.3g}"),
                   "nemenyi_cd_alpha_0.05": round(cd, 3)}]).to_csv(tab / "stats_friedman.csv", index=False)
    pd.DataFrame({"strategy": mean_ranks.index, "mean_rank": mean_ranks.values.round(3),
                  "mean_macro_f1": blocks[mean_ranks.index].mean().values.round(3)}).to_csv(
        tab / "stats_mean_ranks.csv", index=False)
    nem.map(lambda v: float(f"{v:.3g}")).rename_axis("strategy").to_csv(tab / "stats_nemenyi_pvalues.csv")
    sig_pairs = [(a, b, nem.loc[a, b]) for a, b in combinations(STRATS, 2) if nem.loc[a, b] < 0.05]

    # Planned pairwise tests: each strategy vs FedAvg, Wilcoxon signed-rank over the same 12
    # blocks, two-sided, Holm correction over the 6 comparisons. Effect sizes: median difference
    # (strategy - FedAvg) and matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-), zeros dropped.
    wrows = []
    for s in [x for x in STRATS if x != "fedavg"]:
        diff = (blocks[s] - blocks["fedavg"]).to_numpy()
        nz = diff[diff != 0]
        rk = rankdata(np.abs(nz))
        w_pos, w_neg = rk[nz > 0].sum(), rk[nz < 0].sum()
        test = wilcoxon(blocks[s], blocks["fedavg"], alternative="two-sided", zero_method="wilcox")
        wrows.append({"strategy": s, "baseline": "fedavg", "n_blocks": len(diff), "n_nonzero": len(nz),
                      "n_better": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                      "median_diff": np.median(diff), "mean_diff": diff.mean(),
                      "w_plus": w_pos, "w_minus": w_neg, "wilcoxon_statistic": float(test.statistic),
                      "p_raw": float(test.pvalue),
                      "rank_biserial_r": (w_pos - w_neg) / (w_pos + w_neg) if len(nz) else 0.0})
    wt = pd.DataFrame(wrows)
    order_p = np.argsort(wt.p_raw.to_numpy())           # Holm step-down
    m = len(wt)
    adj, running = np.empty(m), 0.0
    for rank_i, idx in enumerate(order_p):
        running = max(running, min(1.0, (m - rank_i) * wt.p_raw.iloc[idx]))
        adj[idx] = running
    wt["p_holm"] = adj
    wt["significant_holm_0.05"] = wt.p_holm < 0.05
    for c in ("median_diff", "mean_diff", "w_plus", "w_minus", "wilcoxon_statistic", "rank_biserial_r"):
        wt[c] = wt[c].round(3)
    for c in ("p_raw", "p_holm"):
        wt[c] = wt[c].map(lambda v: float(f"{v:.3g}"))
    wt["protocol"], wt["rounds"], wt["metric"] = "random", 100, "val-selected test macro-F1"
    wt["blocks"] = "partition x seed (4 x 3)"
    wt["note"] = ("blocks sharing a partition are not fully independent (same partition scheme, "
                  "different seeds), so p-values are approximate; diff = strategy - fedavg")
    wt.to_csv(tab / "stats_wilcoxon_vs_fedavg.csv", index=False)

    # CD diagram (Demsar style): rank axis, CD bar, cliques = maximal runs with all pairs p >= 0.05
    order = list(mean_ranks.index)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and all(nem.loc[order[a], order[b]] >= 0.05 for a in range(i, j + 2) for b in range(i, j + 2) if a < b):
            j += 1
        if j > i and not any(c0 <= i and j <= c1 for c0, c1 in cliques):
            cliques.append((i, j))
    fig, ax = plt.subplots(figsize=(SINGLE, 1.75))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0.2, 1.08)
    ax.axis("off")
    y_axis = 0.74
    ax.plot([lo, hi], [y_axis, y_axis], color="k", lw=0.8)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y_axis, y_axis + 0.03], color="k", lw=0.8)
        ax.text(t, y_axis + 0.05, str(t), ha="center", va="bottom")
    ax.plot([lo, lo + cd], [0.96, 0.96], color="k", lw=1.2)
    for xx in (lo, lo + cd):
        ax.plot([xx, xx], [0.94, 0.98], color="k", lw=1.0)
    ax.text(lo + cd + 0.1, 0.96, f"CD = {cd:.2f}", ha="left", va="center")
    half = (k + 1) // 2
    for idx, s in enumerate(order):
        r = mean_ranks[s]
        left = idx < half
        y = y_axis - 0.16 - 0.12 * (idx if left else (k - 1 - idx))
        xt = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xt], [y_axis, y, y], color=COLOR[s] if s != "trimmed_mean" else "#B8A800", lw=0.9)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{LABEL[s]} ({r:.2f})", ha="right" if left else "left", va="center")
    for ci, (i, j) in enumerate(cliques):
        yy = y_axis - 0.04 - 0.035 * ci
        ax.plot([mean_ranks[order[i]] - 0.05, mean_ranks[order[j]] + 0.05], [yy, yy], color="k", lw=2.2)
    save(fig, figd / "fig6_cd_diagram.png")

    # ---------------------------------------------------------------- B. figures
    # fig1 leakage diagnostic
    fig, ax = plt.subplots(figsize=(SINGLE, 2.1))
    cats = ["Macro-F1"] + CLASSES
    x = np.arange(len(cats))
    w = 0.38
    for i, (pname, col, hatch) in enumerate((("chronological", "#555555", ""), ("random", "#BBBBBB", "///"))):
        row = t_leak[t_leak.protocol == pname].iloc[0]
        means = [row["macro_f1_mean"]] + [row[f"recall_{c}_mean"] for c in CLASSES]
        sds = [row["macro_f1_std"]] + [row[f"recall_{c}_std"] for c in CLASSES]
        ax.bar(x + (i - 0.5) * w, means, w, yerr=clip_err(means, sds), color=col, hatch=hatch, edgecolor="k", lw=0.4,
               capsize=1.5, error_kw={"lw": 0.6}, label=f"{pname.capitalize()} split")
    ax.set_xticks(x, ["Macro-\nF1", "Normal", "BFA", "DDoS", "DoS", "Probe"])
    ax.set_ylabel("Test score (centralized)")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
    save(fig, figd / "fig1_leakage_diagnostic.png")

    # fig2 heatmap (double column)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.3), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (pname, d, lab) in zip(axes, (("random", rnd, "(a) Random split, R = 100"),
                                           ("chronological", chrono, "(b) Chronological split, R = 30"))):
        th = t_head[(t_head.protocol == pname) & (t_head.strategy != "centralized")]
        M = th.pivot(index="strategy", columns="partition", values="macro_f1_mean").loc[STRATS, PARTS]
        S = th.pivot(index="strategy", columns="partition", values="macro_f1_std").loc[STRATS, PARTS]
        im = ax.imshow(M.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for i in range(len(STRATS)):
            for j in range(len(PARTS)):
                v = M.iat[i, j]
                ax.text(j, i, f"{v:.2f}\n±{S.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="k" if v > 0.6 else "w", linespacing=0.9)
        ax.set_xticks(range(len(PARTS)), [PLABEL[p].replace(" (", "\n(") for p in PARTS])
        ax.set_yticks(range(len(STRATS)), [LABEL[s] for s in STRATS])
        ax.tick_params(length=0)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        cm, cs = ms(d["central"].sel_macro_f1)
        ax.text(0.5, -0.2, f"{lab}\nCentralized: {cm:.3f} ± {cs:.3f}", transform=ax.transAxes,
                ha="center", va="top")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Val-selected test macro-F1")
    save(fig, figd / "fig2_heatmap_macro_f1.png")

    # fig3 convergence (random, R = 100, P0 and P3)
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 4.2), sharex=True)
    cm = rnd["central"].sel_macro_f1.mean()
    for ax, p in zip(axes, ("P0", "P3_m1")):
        for s in STRATS:
            ids = rnd["clean"][(rnd["clean"].strategy == s) & (rnd["clean"].partition == p)].run_id
            g = rnd["rounds"][rnd["rounds"].run_id.isin(ids)].groupby("round")["macro_f1"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COLOR[s], ls=LSTYLE[s], lw=1.1, label=LABEL[s])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[s], alpha=0.15, lw=0)
        ax.axhline(cm, color="k", ls="--", lw=0.9, label="Centralized")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test macro-F1")
        ax.text(0.98, 0.04, PLABEL[p], transform=ax.transAxes, ha="right", va="bottom")
    axes[1].set_xlabel("Round")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)
    fig.tight_layout(h_pad=0.6)
    save(fig, figd / "fig3_convergence_random_r100.png")

    # fig4 per-class recall under random P3
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    cls_order = ["DoS", "Probe", "DDoS", "BFA", "Normal"]
    x = np.arange(len(cls_order))
    w = 0.8 / len(STRATS)
    g3 = rnd["clean"][rnd["clean"].partition == "P3_m1"]
    for i, s in enumerate(STRATS):
        g = g3[g3.strategy == s]
        means = [g[f"sel_recall_{c}"].mean() for c in cls_order]
        sds = [g[f"sel_recall_{c}"].std(ddof=1) for c in cls_order]
        ax.bar(x - 0.4 + (i + 0.5) * w, means, w, yerr=clip_err(means, sds), color=COLOR[s], edgecolor="k", lw=0.3,
               capsize=0.8, error_kw={"lw": 0.5}, label=LABEL[s])
    ax.set_xticks(x, cls_order)
    ax.set_ylabel("Test recall (random, P3)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4, columnspacing=0.8, handlelength=1.2)
    save(fig, figd / "fig4_per_class_recall_random_P3.png")

    # fig5 poisoning
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.6), sharex=True)
    conds = ["clean", "sign_flip", "targeted"]
    pstr = ["fedavg", "krum", "median", "trimmed_mean"]
    x = np.arange(len(conds))
    w = 0.8 / len(pstr)
    for ax, (col, ylab) in zip(axes, (("macro_f1", "Test macro-F1"), ("attack_fnr", "Attack FNR"))):
        for i, s in enumerate(pstr):
            g = t_poison[t_poison.strategy == s].set_index("attack").loc[conds]
            ax.bar(x - 0.4 + (i + 0.5) * w, g[f"{col}_mean"], w,
                   yerr=clip_err(g[f"{col}_mean"], g[f"{col}_std"]), color=COLOR[s],
                   edgecolor="k", lw=0.3, capsize=1.2, error_kw={"lw": 0.5}, label=LABEL[s])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1)
    axes[1].set_xticks(x, ["Clean", "Sign flip", "Targeted"])
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(h_pad=0.5)
    save(fig, figd / "fig5_poisoning.png")

    # console summary
    print(f"Friedman chi2 = {stat:.3f}, p = {pval:.3g} (k = {k}, n = {n}); Nemenyi CD(0.05) = {cd:.3f}")
    print("Mean ranks (1 = best):", ", ".join(f"{s} {r:.2f}" for s, r in mean_ranks.items()))
    print("Significant pairs (p < 0.05):", ", ".join(f"{a}-{b} ({p:.3g})" for a, b, p in sig_pairs) or "none")
    print("Cliques (not significantly different):",
          [[order[t] for t in range(i, j + 1)] for i, j in cliques])


if __name__ == "__main__":
    main()
