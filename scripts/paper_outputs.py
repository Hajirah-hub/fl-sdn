"""Paper statistics, figures and tables from the FINAL v2 results (never results/dry*).

    python scripts/paper_outputs.py

Inputs ($FEDSDN_RESULTS, written by scripts/run_study.py): tuning/, random/ (R = 100),
chrono/ (R = 30, incl. poisoning on P4), sweep/ (R = 100, FedAvg / FedAdam on P1 a0.1/1/10 and
P2 b0.5), local/ (local-only baseline), poison_random/ (random P3 m1, R = 100), plus the
partition sidecars in $FEDSDN_DATA/partitions/. Outputs: results/paper_tables/*.csv and
results/paper_figures/*.png (300 dpi, IEEE single column 3.5 in unless noted, fonts >= 8 pt,
Okabe-Ito palette with a fixed color per strategy, SCAFFOLD grey, centralized black, no titles).

Derived without extra runs: a round never depends on R, so an R = 30 run is exactly the first
30 rounds of the R = 100 run with the same seed. The R = 30 vs R = 100 table and the equal-budget
leakage diagnostic (centralized, 30 epochs on both protocols) truncate the R = 100 rounds and
re-select the round by validation macro-F1 (ties -> earliest).

Statistics (random protocol, R = 100, val-selected test macro-F1, blocks = partition x seed =
4 x 5 = 20, 8 strategies): Friedman + Nemenyi (alpha 0.05); Wilcoxon signed-rank of each
strategy vs FedAvg, two-sided, Holm over 7 comparisons, effect sizes median difference and
matched-pairs rank-biserial r; bootstrap 95% CIs of the mean difference to FedAvg, stratified by
partition (seeds resampled within each partition, 10,000 resamples, percentile, rng 0); OLS per
strategy of the gap to centralized (centralized val-selected F1 of the same seed minus the run's)
on the partition's realized mean pairwise JSD and shard-size CV, using the main grid plus the
alpha sweep for FedAvg / FedAdam. Blocks sharing a partition are not fully independent, so
p-values are approximate. Rounding: 3 decimals; p-values 3 significant digits.
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
import statsmodels.api as sm  # noqa: E402
from scipy.stats import friedmanchisquare, rankdata, wilcoxon  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_reduced import add_t95  # noqa: E402

STRATS = ["fedavg", "fedprox", "fednova", "fedadam", "scaffold", "krum", "median", "trimmed_mean"]
LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "fednova": "FedNova", "fedadam": "FedAdam",
         "scaffold": "SCAFFOLD", "krum": "Krum", "median": "Median", "trimmed_mean": "Trimmed mean",
         "central": "Centralized"}
# Okabe-Ito palette, one fixed color per strategy; SCAFFOLD grey; centralized = black
COLOR = {"fedavg": "#0072B2", "fedprox": "#56B4E9", "fednova": "#009E73", "fedadam": "#D55E00",
         "scaffold": "#999999", "krum": "#CC79A7", "median": "#E69F00", "trimmed_mean": "#F0E442",
         "central": "#000000"}
LSTYLE = {"fedavg": "-", "fedprox": "--", "fednova": "-", "fedadam": "-", "scaffold": (0, (5, 1.5)),
          "krum": "-.", "median": "-", "trimmed_mean": ":", "central": "--"}
PARTS = ["P0", "P1_a0.5", "P3_m1", "P4_a0.5b0.5"]
SWEEP = ["P1_a0.1", "P1_a1", "P1_a10", "P2_b0.5"]
PLABEL = {"P0": "P0 (IID)", "P1_a0.5": "P1 (α=0.5)", "P3_m1": "P3 (m=1)", "P4_a0.5b0.5": "P4 (α=β=0.5)",
          "P1_a0.1": "P1 (α=0.1)", "P1_a1": "P1 (α=1)", "P1_a10": "P1 (α=10)", "P2_b0.5": "P2 (β=0.5)"}
CLASSES = ["Normal", "BFA", "DDoS", "DoS", "Probe"]
P4, P3 = "P4_a0.5b0.5", "P3_m1"
POISON = ["fedavg", "krum", "median", "trimmed_mean"]
SEEDS = [0, 1, 2, 3, 4]
SINGLE, DOUBLE = 3.5, 7.16
Q05_K8 = 3.031            # Nemenyi critical value q_0.05 for k = 8 (Demsar 2006, Table 5)
N_BOOT = 10_000


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "axes.linewidth": 0.6, "lines.linewidth": 1.2, "savefig.dpi": 300, "figure.dpi": 100,
        "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    })


def r3(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(3)


def p3s(v: float) -> float:
    return float(f"{v:.3g}")


def load(d: Path) -> dict:
    s = pd.read_csv(d / "summary.csv")
    r = pd.read_csv(d / "rounds.csv")
    central = s[s.kind == "central"]
    thr = 0.95 * central["sel_macro_f1"].mean() if len(central) else np.nan
    fed = s[s.kind == "fed"]
    clean = fed[fed.attack == "none"]
    thr_stable = np.nan
    if len(central):
        thr_stable = 0.95 * stable(central)["sel_macro_f1"].mean()
        st = add_t95(clean, r, thr_stable)
        clean = add_t95(clean, r, thr)
        clean["t95_stable"], clean["t95_stable_censored"] = st["t95"].to_numpy(), st["t95_censored"].to_numpy()
    return {"summary": s, "rounds": r, "central": central, "clean": clean, "thr": thr,
            "thr_stable": thr_stable, "R": int(s["R"].max()), "poison": fed[fed.attack != "none"]}


def ms(x: pd.Series) -> tuple[float, float]:
    x = pd.Series(x).dropna()
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0


def clip_err(means, sds) -> np.ndarray:
    """Asymmetric error bars: mean +/- std, clipped to the [0, 1] score range."""
    m, sd = np.asarray(means, float), np.nan_to_num(np.asarray(sds, float))
    return np.vstack([np.minimum(sd, m), np.minimum(sd, 1 - m)])


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def truncate_select(rounds: pd.DataFrame, run_ids, R: int) -> pd.DataFrame:
    """Per run: the round <= R with the best validation macro-F1 (ties -> earliest)."""
    out = []
    for rid in run_ids:
        rr = rounds[(rounds.run_id == rid) & (rounds["round"] <= R)].sort_values("round")
        out.append(rr.iloc[int(np.argmax(rr["val_macro_f1"].to_numpy()))])
    return pd.DataFrame(out).reset_index(drop=True)


def sidecar(data_dir: Path, proto_dir: str, part: str, seed: int) -> dict:
    return json.loads((data_dir / "partitions" / proto_dir / f"{part}_K8_s{seed}.json").read_text())


def holm(p: np.ndarray) -> np.ndarray:
    order, m = np.argsort(p), len(p)
    adj, running = np.empty(m), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def wilcoxon_rows(blocks: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank (a - b) over the blocks; Holm over the given pairs.
    Effect sizes: median / mean difference and matched-pairs rank-biserial r (zeros dropped)."""
    rows = []
    for a, b in pairs:
        diff = (blocks[a] - blocks[b]).to_numpy()
        nz = diff[diff != 0]
        rk = rankdata(np.abs(nz))
        w_pos, w_neg = rk[nz > 0].sum(), rk[nz < 0].sum()
        test = wilcoxon(blocks[a], blocks[b], alternative="two-sided", zero_method="wilcox")
        rows.append({"strategy": a, "baseline": b, "n_blocks": len(diff), "n_nonzero": len(nz),
                     "n_better": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                     "median_diff": np.median(diff), "mean_diff": diff.mean(),
                     "w_plus": w_pos, "w_minus": w_neg, "wilcoxon_statistic": float(test.statistic),
                     "p_raw": float(test.pvalue),
                     "rank_biserial_r": (w_pos - w_neg) / (w_pos + w_neg) if len(nz) else 0.0})
    wt = pd.DataFrame(rows)
    wt["p_holm"] = holm(wt.p_raw.to_numpy())
    wt["significant_holm_0.05"] = wt.p_holm < 0.05
    for c in ("median_diff", "mean_diff", "w_plus", "w_minus", "wilcoxon_statistic", "rank_biserial_r"):
        wt[c] = wt[c].round(3)
    for c in ("p_raw", "p_holm"):
        wt[c] = wt[c].map(p3s)
    return wt


def bootstrap_rows(blocks: pd.DataFrame, pairs: list[tuple[str, str]]) -> pd.DataFrame:
    """95% percentile CI of the mean difference (a - b); seeds resampled within each partition."""
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(SEEDS), size=(N_BOOT, len(PARTS), len(SEEDS)))
    rows = []
    for a, b in pairs:
        d = (blocks[a] - blocks[b]).unstack("seed").loc[PARTS, SEEDS].to_numpy()          # 4 x 5
        boot = np.take_along_axis(np.broadcast_to(d, (N_BOOT, *d.shape)), idx, axis=2).mean(axis=(1, 2))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"strategy": a, "baseline": b, "mean_diff": d.mean(), "ci95_low": lo, "ci95_high": hi,
                     "ci_excludes_zero": bool(lo > 0 or hi < 0), "n_boot": N_BOOT,
                     "resampling": "seeds within partition (stratified), percentile"})
    return pd.DataFrame(rows)


def stable(central: pd.DataFrame) -> pd.DataFrame:
    """Centralized runs that neither diverged nor collapsed by their final epoch."""
    return central[~central.diverged.astype(bool) & ~central.collapsed.astype(bool)]


def main() -> None:
    res = Path(os.environ.get("FEDSDN_RESULTS", "./results"))
    data_dir = Path(os.environ.get("FEDSDN_DATA", "./data"))
    tab, figd = res / "paper_tables", res / "paper_figures"
    tab.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    style()

    rnd, chrono = load(res / "random"), load(res / "chrono")
    sweep, prand = load(res / "sweep"), load(res / "poison_random")
    local = pd.read_csv(res / "local" / "summary.csv")
    tun_trials = pd.read_csv(res / "tuning" / "trials.csv")
    tun_chosen = pd.read_csv(res / "tuning" / "tuning_summary.csv")
    assert rnd["R"] == 100 and chrono["R"] == 30 and sweep["R"] == 100
    assert len(rnd["clean"]) == len(STRATS) * len(PARTS) * len(SEEDS) == len(chrono["clean"])
    protos = {"random": rnd, "chronological": chrono}
    c_rnd = rnd["central"].set_index("seed")["sel_macro_f1"]

    def cell(d, s, p):
        c = d["clean"]
        return c[(c.strategy == s) & (c.partition == p)]

    # ---------------------------------------------------------------- tables: main grid
    rows = []
    for pname, d in protos.items():
        for s in STRATS:
            for p in PARTS:
                m, sd = ms(cell(d, s, p).sel_macro_f1)
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "macro_f1_mean": m, "macro_f1_std": sd, "n_seeds": len(cell(d, s, p))})
        m, sd = ms(d["central"].sel_macro_f1)
        rows.append({"protocol": pname, "rounds": d["R"], "strategy": "centralized", "partition": "all",
                     "macro_f1_mean": m, "macro_f1_std": sd, "n_seeds": len(d["central"])})
        st = stable(d["central"])
        m, sd = ms(st.sel_macro_f1)
        rows.append({"protocol": pname, "rounds": d["R"], "strategy": "centralized (stable seeds)",
                     "partition": "seeds " + " ".join(map(str, sorted(st.seed))),
                     "macro_f1_mean": m, "macro_f1_std": sd, "n_seeds": len(st)})
    t_head = pd.DataFrame(rows)
    r3(t_head).to_csv(tab / "table_valselected_macro_f1.csv", index=False)

    def grp(d, s, p):
        return d["central"] if s == "centralized" else cell(d, s, p)

    rows, rows_cls, rows_worst, rows_t95 = [], [], [], []
    for pname, d in protos.items():
        for s in STRATS + ["centralized"]:
            for p in (PARTS if s != "centralized" else ["all"]):
                g = grp(d, s, p)
                fm, fs = ms(g.final_macro_f1)
                am, as_ = ms(g.auc_macro_f1)
                sm_, ss = ms(g.sel_round)
                rows.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                             "final_macro_f1_mean": fm, "final_macro_f1_std": fs,
                             "auc_macro_f1_mean": am, "auc_macro_f1_std": as_,
                             "selected_round_mean": sm_, "selected_round_std": ss})
                for c in CLASSES:
                    m, sd = ms(g[f"sel_recall_{c}"])
                    rows_cls.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                                     "class": c, "recall_mean": m, "recall_std": sd})
                if s == "centralized":
                    continue
                m, sd = ms(g.sel_worst_client_val_f1)
                rows_worst.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                                   "worst_client_val_macro_f1_mean": m, "worst_client_val_macro_f1_std": sd})
                bpr = int(g.bytes_per_round.iloc[0])
                n_st = len(stable(d["central"]))
                for ref, col, thr_ in ((f"all {len(d['central'])} seeds", "t95", d["thr"]),
                                       (f"stable {n_st} seeds", "t95_stable", d["thr_stable"])):
                    hit = g[~g[f"{col}_censored"]][col]
                    rows_t95.append({"protocol": pname, "rounds": d["R"], "strategy": s, "partition": p,
                                     "centralized_reference": ref, "t95_threshold": thr_, "n_reached": len(hit),
                                     "n_censored": int(g[f"{col}_censored"].sum()),
                                     "t95_rounds_mean_reached": hit.mean() if len(hit) else np.nan,
                                     "t95_rounds_std_reached": hit.std(ddof=1) if len(hit) > 1 else (0.0 if len(hit) else np.nan),
                                     "mb_to_t95_mean_reached": hit.mean() * bpr / 1e6 if len(hit) else np.nan,
                                     "bytes_per_round": bpr, "mb_per_round": bpr / 1e6})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_final_auc_macro_f1.csv", index=False)
    r3(pd.DataFrame(rows_cls)).to_csv(tab / "table_per_class_recall.csv", index=False)
    r3(pd.DataFrame(rows_worst)).to_csv(tab / "table_worst_client_val_f1.csv", index=False)
    r3(pd.DataFrame(rows_t95)).to_csv(tab / "table_t95_bytes.csv", index=False)

    # ---------------------------------------------------------------- poisoning
    def poison_table(d, part, pname):
        out = []
        for s in POISON:
            conds = {"clean": cell(d, s, part) if d is not prand else cell(rnd, s, part)}
            for a in ("sign_flip", "targeted"):
                conds[a] = d["poison"][(d["poison"].strategy == s) & (d["poison"].attack == a)]
            for a, g in conds.items():
                fm, fs = ms(g.sel_macro_f1)
                nm, ns = ms(g.sel_attack_fnr)
                row = {"protocol": pname, "rounds": d["R"], "partition": part, "strategy": s, "attack": a,
                       "malicious_clients": 0 if a == "clean" else 2, "n_seeds": len(g),
                       "macro_f1_mean": fm, "macro_f1_std": fs, "attack_fnr_mean": nm, "attack_fnr_std": ns}
                if a != "clean" and "malicious_sole_holder" in g:
                    sole = g.malicious_sole_holder.astype(bool)
                    row["n_runs_malicious_sole_holder"] = int(sole.sum())
                    row["sole_holder_families"] = "; ".join(
                        f"s{r.seed}:{r.sole_holder_families}" for r in g[sole].sort_values("seed").itertuples())
                    row["macro_f1_mean_sole_holder"] = g[sole].sel_macro_f1.mean() if sole.any() else np.nan
                    row["macro_f1_mean_not_sole_holder"] = g[~sole].sel_macro_f1.mean() if (~sole).any() else np.nan
                out.append(row)
        return pd.DataFrame(out)

    t_poison = poison_table(chrono, P4, "chronological")
    r3(t_poison.drop(columns=[c for c in t_poison if "sole" in c])).to_csv(tab / "table_poisoning.csv", index=False)
    t_poison_r = poison_table(prand, P3, "random")
    r3(t_poison_r).to_csv(tab / "table_poisoning_random_P3.csv", index=False)

    # ---------------------------------------------------------------- R = 30 vs 100, leakage
    rows = []
    thr_r30 = 0.95 * truncate_select(rnd["rounds"], rnd["central"].run_id, 30)["macro_f1"].mean()
    for s in STRATS:
        for p in PARTS:
            b = cell(rnd, s, p)
            a = truncate_select(rnd["rounds"], b.run_id, 30)
            am, as_ = ms(a.macro_f1)
            bm, bs = ms(b.sel_macro_f1)
            rr = rnd["rounds"][rnd["rounds"].run_id.isin(b.run_id) & (rnd["rounds"]["round"] <= 30)]
            r30_hit = rr[rr.macro_f1 >= rnd["thr"]].run_id.nunique()
            rows.append({"protocol": "random", "strategy": s, "partition": p,
                         "r30_macro_f1_mean": am, "r30_macro_f1_std": as_,
                         "r100_macro_f1_mean": bm, "r100_macro_f1_std": bs,
                         "r30_selected_round_mean": a["round"].mean(),
                         "r100_selected_round_mean": b.sel_round.mean(),
                         "r100_selected_round_min": b.sel_round.min(), "r100_selected_round_max": b.sel_round.max(),
                         "t95_threshold_r100": rnd["thr"], "t95_threshold_central_30_epochs": thr_r30,
                         "r30_t95_n_reached": r30_hit, "r100_t95_n_reached": int((~b.t95_censored).sum())})
    r3(pd.DataFrame(rows)).to_csv(tab / "table_random_r30_vs_r100.csv", index=False)

    rows = []
    for pname, g in (("chronological", chrono["central"].rename(columns=lambda c: c[4:] if c.startswith("sel_") else c)),
                     ("random", truncate_select(rnd["rounds"], rnd["central"].run_id, 30))):
        row = {"protocol": pname, "epochs": 30, "n_seeds": len(g)}
        row["macro_f1_mean"], row["macro_f1_std"] = ms(g.macro_f1)
        for c in CLASSES:
            row[f"recall_{c}_mean"], row[f"recall_{c}_std"] = ms(g[f"recall_{c}"])
        rows.append(row)
    t_leak = pd.DataFrame(rows)
    r3(t_leak).to_csv(tab / "table_leakage_diagnostic.csv", index=False)

    # ---------------------------------------------------------------- heterogeneity
    rows = []
    for pdir, pname, parts, seeds in (("insdn", "chronological", PARTS, SEEDS),
                                      ("insdn_random", "random", PARTS + SWEEP, SEEDS),
                                      ("insdn_random", "random (tuning)", ["P1_a0.5", P3, P4], [99])):
        for p in parts:
            for seed in seeds:
                sc = sidecar(data_dir, pdir, p, seed)
                rows.append({"protocol": pname, "partition": p, "seed": seed, "K": sc["K"],
                             "mean_pairwise_jsd": sc["mean_pairwise_jsd"], "shard_size_cv": sc["shard_size_cv"],
                             "min_shard": sc["min_shard"], "max_shard": sc["max_shard"],
                             "attempts": sc["attempts"], "rows_moved": sc["rows_moved"]})
    het = pd.DataFrame(rows)
    r3(het).to_csv(tab / "table_partition_heterogeneity.csv", index=False)
    het_r = het[het.protocol == "random"].set_index(["partition", "seed"])

    # ---------------------------------------------------------------- tuning
    tt = tun_trials.copy()
    tt["label"] = tt.strategy.map(LABEL)
    r3(tt).to_csv(tab / "table_tuning_trials.csv", index=False)
    r3(tun_chosen).to_csv(tab / "table_tuning_chosen.csv", index=False)

    # ---------------------------------------------------------------- alpha sweep
    rows = []
    for s in ("fedavg", "fedadam"):
        for p in ["P1_a0.1", "P1_a0.5", "P1_a1", "P1_a10", "P0", "P2_b0.5"]:
            g = cell(sweep if p in SWEEP else rnd, s, p)
            m, sd = ms(g.sel_macro_f1)
            fm, fs = ms(g.final_macro_f1)
            h = het_r.loc[p]
            rows.append({"protocol": "random", "rounds": 100, "strategy": s, "partition": p,
                         "mean_jsd": h.mean_pairwise_jsd.mean(), "mean_cv": h.shard_size_cv.mean(),
                         "macro_f1_mean": m, "macro_f1_std": sd, "final_macro_f1_mean": fm,
                         "final_macro_f1_std": fs, "n_seeds": len(g)})
    t_sweep = pd.DataFrame(rows)
    r3(t_sweep).to_csv(tab / "table_alpha_sweep.csv", index=False)

    # ---------------------------------------------------------------- local-only
    rows = []
    for p in PARTS:
        g = local[local.partition == p]
        row = {"protocol": "random", "epochs": 100, "partition": p, "n_seeds": len(g)}
        for col, name in (("sel_macro_f1", "local_mean_client"), ("sel_worst_macro_f1", "local_worst_client"),
                          ("sel_best_macro_f1", "local_best_client")):
            row[f"{name}_mean"], row[f"{name}_std"] = ms(g[col])
        row["classes_per_client_mean"] = np.mean([np.mean(list(map(int, x.split()))) for x in g.client_n_classes])
        for s in ("fedavg", "fedadam"):
            row[f"{s}_mean"], row[f"{s}_std"] = ms(cell(rnd, s, p).sel_macro_f1)
        row["centralized_mean"], row["centralized_std"] = ms(rnd["central"].sel_macro_f1)
        rows.append(row)
    t_local = pd.DataFrame(rows)
    r3(t_local).to_csv(tab / "table_local_only.csv", index=False)

    # ---------------------------------------------------------------- statistics
    cl = rnd["clean"]
    blocks = cl.pivot_table(index=["partition", "seed"], columns="strategy", values="sel_macro_f1")[STRATS]
    assert blocks.shape == (20, 8) and not blocks.isna().any().any()
    stat, pval = friedmanchisquare(*[blocks[s].to_numpy() for s in STRATS])
    ranks = blocks.rank(axis=1, ascending=False)          # rank 1 = highest macro-F1
    mean_ranks = ranks.mean().sort_values()
    k, n = len(STRATS), len(blocks)
    cd = Q05_K8 * np.sqrt(k * (k + 1) / (6 * n))
    nem = sp.posthoc_nemenyi_friedman(blocks.to_numpy())
    nem.index = nem.columns = STRATS
    pd.DataFrame([{"test": "Friedman", "protocol": "random", "rounds": 100, "metric": "val-selected test macro-F1",
                   "k_strategies": k, "n_blocks": n, "blocks": "partition x seed (4 x 5)",
                   "chi2_statistic": round(stat, 3), "p_value": p3s(pval),
                   "nemenyi_cd_alpha_0.05": round(cd, 3)}]).to_csv(tab / "stats_friedman.csv", index=False)
    pd.DataFrame({"strategy": mean_ranks.index, "mean_rank": mean_ranks.values.round(3),
                  "mean_macro_f1": blocks[mean_ranks.index].mean().values.round(3)}).to_csv(
        tab / "stats_mean_ranks.csv", index=False)
    nem.map(p3s).rename_axis("strategy").to_csv(tab / "stats_nemenyi_pvalues.csv")
    sig_pairs = [(a, b, nem.loc[a, b]) for a, b in combinations(STRATS, 2) if nem.loc[a, b] < 0.05]

    others = [x for x in STRATS if x != "fedavg"]
    wt = wilcoxon_rows(blocks, [(x, "fedavg") for x in others])
    wt["protocol"], wt["rounds"], wt["metric"] = "random", 100, "val-selected test macro-F1"
    wt["blocks"] = "partition x seed (4 x 5)"
    wt["note"] = ("blocks sharing a partition are not fully independent (same partition scheme, "
                  "different seeds), so p-values are approximate; diff = strategy - fedavg")
    wt.to_csv(tab / "stats_wilcoxon_vs_fedavg.csv", index=False)
    bt = bootstrap_rows(blocks, [(x, "fedavg") for x in others])
    r3(bt).to_csv(tab / "stats_bootstrap_ci_vs_fedavg.csv", index=False)

    # regression of the gap to centralized on realized JSD and CV
    reg_rows, reg_points = [], []
    stable_seeds = sorted(stable(rnd["central"]).seed)
    for (ref, seeds_ok), s in [((r_, k_), s_) for r_, k_ in ((f"all {len(c_rnd)} seeds", list(c_rnd.index)),
                                                            (f"stable {len(stable_seeds)} seeds", stable_seeds))
                               for s_ in STRATS]:
        g = pd.concat([cl[cl.strategy == s], sweep["clean"][sweep["clean"].strategy == s]])
        g = g[g.seed.isin(seeds_ok)]
        h = het_r.loc[list(zip(g.partition, g.seed))]
        X = pd.DataFrame({"jsd": h.mean_pairwise_jsd.to_numpy(), "cv": h.shard_size_cv.to_numpy()})
        y = c_rnd.loc[g.seed].to_numpy() - g.sel_macro_f1.to_numpy()
        reg_points.append(pd.DataFrame({"centralized_reference": ref, "strategy": s, "partition": g.partition.to_numpy(), "seed": g.seed.to_numpy(),
                                        "jsd": X.jsd, "cv": X.cv, "gap": y}))
        fit = sm.OLS(y, sm.add_constant(X)).fit()
        ci = fit.conf_int()
        reg_rows.append({"centralized_reference": ref, "strategy": s, "n_runs": len(y), "partitions": " ".join(sorted(g.partition.unique())),
                         "intercept": fit.params["const"],
                         "coef_jsd": fit.params["jsd"], "se_jsd": fit.bse["jsd"], "p_jsd": p3s(fit.pvalues["jsd"]),
                         "ci95_jsd_low": ci.loc["jsd", 0], "ci95_jsd_high": ci.loc["jsd", 1],
                         "coef_cv": fit.params["cv"], "se_cv": fit.bse["cv"], "p_cv": p3s(fit.pvalues["cv"]),
                         "ci95_cv_low": ci.loc["cv", 0], "ci95_cv_high": ci.loc["cv", 1],
                         "r2": fit.rsquared, "adj_r2": fit.rsquared_adj, "corr_jsd_cv": X.jsd.corr(X.cv),
                         "mean_gap": y.mean()})
    rt = pd.DataFrame(reg_rows)
    rt["note"] = ("gap = centralized val-selected test macro-F1 (same seed) - run's; OLS gap ~ 1 + JSD + CV; "
                  "random protocol R = 100; fedavg/fedadam include the alpha sweep; 'stable' drops runs whose "
                  "same-seed centralized run collapsed")
    r3(rt).to_csv(tab / "stats_regression_gap.csv", index=False)
    r3(pd.concat(reg_points)).to_csv(tab / "stats_regression_points.csv", index=False)

    # ---------------------------------------------------------------- divergence / collapse
    rows = []
    for suite in ("random", "chrono", "sweep", "poison_random", "lr_sensitivity", "tuning", "lr_boundary"):
        f = res / suite / "summary.csv"
        if not f.exists():
            continue
        sm_ = pd.read_csv(f)
        for c in ("diverged", "collapsed"):
            sm_[c] = sm_[c].fillna(False).astype(bool)
        for (proto, st, att), g in sm_.groupby(["protocol", "strategy", "attack"]):
            rows.append({"suite": suite, "protocol": proto, "rounds": int(g.R.max()), "strategy": st, "attack": att,
                         "n_runs": len(g), "n_diverged": int(g.diverged.sum()), "n_collapsed_final": int(g.collapsed.sum()),
                         "diverged_rate": g.diverged.mean(), "collapsed_rate": g.collapsed.mean(),
                         "collapsed_runs": " ".join(g[g.collapsed].run_id),
                         "val_selected_f1_mean_collapsed": g[g.collapsed].sel_macro_f1.mean() if g.collapsed.any() else np.nan,
                         "val_selected_f1_mean_ok": g[~g.collapsed].sel_macro_f1.mean() if (~g.collapsed).any() else np.nan})
    t_fail = pd.DataFrame(rows)
    t_fail["note"] = ("diverged = non-finite weights; collapsed = at the final round one class has test recall >= 0.99 "
                      "and all others <= 0.01 (predictions nearly all one class; includes diverged runs)")
    r3(t_fail).to_csv(tab / "table_divergence_collapse.csv", index=False)

    # ---------------------------------------------------------------- lr boundary + sensitivity
    if (res / "lr_boundary" / "lr_boundary.csv").exists():
        r3(pd.read_csv(res / "lr_boundary" / "lr_boundary.csv")).to_csv(tab / "table_lr_boundary.csv", index=False)
    sens = None
    if (res / "lr_sensitivity" / "summary.csv").exists():
        sens = pd.read_csv(res / "lr_sensitivity" / "summary.csv")
        sens["strategy"] = sens.strategy + "_lr0.6"
        sb = pd.concat([blocks, sens.pivot_table(index=["partition", "seed"], columns="strategy",
                                                 values="sel_macro_f1")], axis=1)
        assert sb.shape[0] == 20 and not sb.isna().any().any()
        fam_a = [("fedadam", "fedavg_lr0.6"), ("fedadam", "fedprox_lr0.6")]
        fam_b = [("fedavg_lr0.6", "fedavg"), ("fedprox_lr0.6", "fedprox")]
        parts_ = []
        for fam, name in ((fam_a, "FedAdam vs lr-0.6 baselines"), (fam_b, "lr 0.6 vs frozen lr")):
            w_ = wilcoxon_rows(sb, fam).merge(bootstrap_rows(sb, fam)[["strategy", "baseline", "ci95_low", "ci95_high",
                                                                       "ci_excludes_zero"]], on=["strategy", "baseline"])
            w_["family"] = name
            parts_.append(w_)
        ts = pd.concat(parts_)
        ts["note"] = ("sensitivity only (not the headline): FedAvg / FedProx rerun at client lr 0.6, all else frozen; "
                      "random protocol R = 100, 20 blocks; Holm within each family of 2")
        r3(ts).to_csv(tab / "stats_lr_sensitivity.csv", index=False)
        rows = []
        for st in ("fedavg", "fedprox"):
            for p_ in PARTS:
                fz = cell(rnd, st, p_)
                s6 = sens[(sens.strategy == f"{st}_lr0.6") & (sens.partition == p_)]
                rows.append({"strategy": st, "partition": p_,
                             "frozen_lr": float(fz.lr.iloc[0]), "frozen_macro_f1_mean": fz.sel_macro_f1.mean(),
                             "frozen_macro_f1_std": fz.sel_macro_f1.std(ddof=1),
                             "lr0.6_macro_f1_mean": s6.sel_macro_f1.mean(), "lr0.6_macro_f1_std": s6.sel_macro_f1.std(ddof=1),
                             "lr0.6_final_macro_f1_mean": s6.final_macro_f1.mean(),
                             "lr0.6_n_diverged": int(s6.diverged.astype(bool).sum()),
                             "lr0.6_n_collapsed": int(s6.collapsed.astype(bool).sum()),
                             "fedadam_macro_f1_mean": cell(rnd, "fedadam", p_).sel_macro_f1.mean()})
        r3(pd.DataFrame(rows)).to_csv(tab / "table_lr_sensitivity.csv", index=False)

    # ---------------------------------------------------------------- figures
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
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.6), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, (pname, d, lab) in zip(axes, (("random", rnd, "(a) Random split, R = 100"),
                                           ("chronological", chrono, "(b) Chronological split, R = 30"))):
        th = t_head[(t_head.protocol == pname) & ~t_head.strategy.str.startswith("centralized")]
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
        ax.text(0.5, -0.17, f"{lab}\nCentralized: {cm:.3f} ± {cs:.3f}", transform=ax.transAxes,
                ha="center", va="top")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Val-selected test macro-F1")
    save(fig, figd / "fig2_heatmap_macro_f1.png")

    # fig3 convergence (random, R = 100, P0 and P3)
    fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 4.4), sharex=True)
    cm = rnd["central"].sel_macro_f1.mean()
    for ax, p in zip(axes, ("P0", P3)):
        for s in STRATS:
            ids = cell(rnd, s, p).run_id
            g = rnd["rounds"][rnd["rounds"].run_id.isin(ids)].groupby("round")["macro_f1"].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COLOR[s], ls=LSTYLE[s], lw=1.1, label=LABEL[s])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COLOR[s], alpha=0.12, lw=0)
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
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    cls_order = ["DoS", "Probe", "DDoS", "BFA", "Normal"]
    x = np.arange(len(cls_order))
    w = 0.84 / len(STRATS)
    for i, s in enumerate(STRATS):
        g = cell(rnd, s, P3)
        means = [g[f"sel_recall_{c}"].mean() for c in cls_order]
        sds = [g[f"sel_recall_{c}"].std(ddof=1) for c in cls_order]
        ax.bar(x - 0.42 + (i + 0.5) * w, means, w, yerr=clip_err(means, sds), color=COLOR[s], edgecolor="k", lw=0.3,
               capsize=0.7, error_kw={"lw": 0.5}, label=LABEL[s])
    ax.set_xticks(x, cls_order)
    ax.set_ylabel("Test recall (random, P3)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=4, columnspacing=0.8, handlelength=1.2)
    save(fig, figd / "fig4_per_class_recall_random_P3.png")

    # fig5 poisoning (double column): (a) chronological P4, R = 30; (b) random P3, R = 100
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.6), sharex=True, sharey="row",
                             gridspec_kw={"wspace": 0.08})
    conds = ["clean", "sign_flip", "targeted"]
    x = np.arange(len(conds))
    w = 0.8 / len(POISON)
    for j, (tp, lab) in enumerate(((t_poison, "(a) Chronological, P4, R = 30"), (t_poison_r, "(b) Random, P3, R = 100"))):
        for i_row, (col, ylab) in enumerate((("macro_f1", "Test macro-F1"), ("attack_fnr", "Attack FNR"))):
            ax = axes[i_row, j]
            for i, s in enumerate(POISON):
                g = tp[tp.strategy == s].set_index("attack").loc[conds]
                ax.bar(x - 0.4 + (i + 0.5) * w, g[f"{col}_mean"], w,
                       yerr=clip_err(g[f"{col}_mean"], g[f"{col}_std"]), color=COLOR[s],
                       edgecolor="k", lw=0.3, capsize=1.2, error_kw={"lw": 0.5}, label=LABEL[s])
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_ylim(0, 1)
        axes[1, j].set_xticks(x, ["Clean", "Sign flip", "Targeted"])
        axes[1, j].text(0.5, -0.3, lab, transform=axes[1, j].transAxes, ha="center", va="top")
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.04, 1.25), ncol=4, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(h_pad=0.5)
    save(fig, figd / "fig5_poisoning.png")

    # fig6 CD diagram (Demsar style)
    order = list(mean_ranks.index)
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and all(nem.loc[order[a], order[b]] >= 0.05 for a in range(i, j + 2) for b in range(i, j + 2) if a < b):
            j += 1
        if j > i and not any(c0 <= i and j <= c1 for c0, c1 in cliques):
            cliques.append((i, j))
    fig, ax = plt.subplots(figsize=(SINGLE, 1.9))
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0.12, 1.08)
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
    for idx_, s in enumerate(order):
        r = mean_ranks[s]
        left = idx_ < half
        y = y_axis - 0.16 - 0.12 * (idx_ if left else (k - 1 - idx_))
        xt = lo - 0.25 if left else hi + 0.25
        ax.plot([r, r, xt], [y_axis, y, y], color=COLOR[s] if s != "trimmed_mean" else "#B8A800", lw=0.9)
        ax.text(xt + (-0.05 if left else 0.05), y, f"{LABEL[s]} ({r:.2f})", ha="right" if left else "left", va="center")
    for ci, (i, j) in enumerate(cliques):
        yy = y_axis - 0.04 - 0.035 * ci
        ax.plot([mean_ranks[order[i]] - 0.05, mean_ranks[order[j]] + 0.05], [yy, yy], color="k", lw=2.2)
    save(fig, figd / "fig6_cd_diagram.png")

    # fig8 tuning: score per trial vs client learning rate
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for s in ["central"] + STRATS:
        g = tun_trials[tun_trials.strategy == s].sort_values("lr")
        ok = g[g.eligible]
        ax.plot(ok.lr, ok.score, color=COLOR[s], ls=LSTYLE[s], lw=0.9, marker="o", ms=2.2, label=LABEL[s])
        bad = g[~g.eligible]
        if len(bad):
            ax.plot(bad.lr, np.zeros(len(bad)) + 0.02, ls="none", marker="x", ms=3.5, color=COLOR[s])
        best = g[g["rank"] == 1]
        ax.plot(best.lr, best.score, ls="none", marker="*", ms=7, color=COLOR[s], mec="k", mew=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Client learning rate")
    ax.set_ylabel("Tuning score (val. macro-F1)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.36), ncol=3, columnspacing=0.8, handlelength=1.6)
    save(fig, figd / "fig8_tuning.png")

    # fig9 heterogeneity sweep (FedAvg / FedAdam)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    sw_parts = ["P1_a0.1", "P1_a0.5", "P1_a1", "P1_a10", "P0", "P2_b0.5"]
    x = np.arange(len(sw_parts))
    w = 0.36
    for i, s in enumerate(("fedavg", "fedadam")):
        g = t_sweep[t_sweep.strategy == s].set_index("partition").loc[sw_parts]
        ax.bar(x + (i - 0.5) * w, g.macro_f1_mean, w, yerr=clip_err(g.macro_f1_mean, g.macro_f1_std),
               color=COLOR[s], edgecolor="k", lw=0.3, capsize=1.2, error_kw={"lw": 0.5}, label=LABEL[s])
    ax.axhline(cm, color="k", ls="--", lw=0.9, label="Centralized")
    jsd = t_sweep[t_sweep.strategy == "fedavg"].set_index("partition").loc[sw_parts, "mean_jsd"]
    ax.set_xticks(x, [f"{PLABEL[p].replace(' (', chr(10) + '(')}\n{jsd[p]:.2f}" for p in sw_parts])
    ax.set_xlabel("Partition (second line: mean realized JSD)")
    ax.set_ylabel("Val-selected test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3)
    save(fig, figd / "fig9_heterogeneity_sweep.png")

    # fig10 regression coefficients (gap ~ JSD + CV), 95% CI
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 2.4), sharey=True, gridspec_kw={"wspace": 0.12})
    yy = np.arange(len(STRATS))[::-1]
    refs = list(rt.centralized_reference.unique())
    for ax, (v, lab) in zip(axes, (("jsd", "JSD coefficient"), ("cv", "CV coefficient"))):
        for ri, (ref, mk, dy) in enumerate(zip(refs, ("o", "s"), (0.15, -0.15))):
            rr_ = rt[rt.centralized_reference == ref].set_index("strategy")
            for yi, s in zip(yy, STRATS):
                r_ = rr_.loc[s]
                ax.errorbar(r_[f"coef_{v}"], yi + dy, xerr=[[r_[f"coef_{v}"] - r_[f"ci95_{v}_low"]],
                                                            [r_[f"ci95_{v}_high"] - r_[f"coef_{v}"]]],
                            fmt=mk, ms=2.6, mfc=None if ri == 0 else "white",
                            color=COLOR[s] if s != "trimmed_mean" else "#B8A800", elinewidth=0.8, capsize=1.2,
                            label=("centralized: " + ref) if (s == "fedavg" and v == "jsd") else None)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab)
    axes[0].set_yticks(yy, [LABEL[s] for s in STRATS])
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.06), ncol=2, handletextpad=0.3)
    save(fig, figd / "fig10_gap_regression.png")

    # fig11 bootstrap CI of difference to FedAvg
    fig, ax = plt.subplots(figsize=(SINGLE, 2.2))
    yy = np.arange(len(others))[::-1]
    sig = wt.set_index("strategy")["significant_holm_0.05"]
    for yi, s in zip(yy, others):
        r_ = bt.set_index("strategy").loc[s]
        ax.errorbar(r_.mean_diff, yi, xerr=[[r_.mean_diff - r_.ci95_low], [r_.ci95_high - r_.mean_diff]], fmt="o",
                    ms=3.5, color=COLOR[s] if s != "trimmed_mean" else "#B8A800", elinewidth=1.0, capsize=2)
        if sig[s]:
            ax.text(r_.ci95_high + 0.01, yi, "*", va="center", ha="left")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(yy, [LABEL[s] for s in others])
    ax.set_xlabel("Macro-F1 difference to FedAvg (95% CI)")
    save(fig, figd / "fig11_bootstrap_ci_vs_fedavg.png")

    # fig12 local-only baseline vs federated
    fig, ax = plt.subplots(figsize=(SINGLE, 2.3))
    x = np.arange(len(PARTS))
    w = 0.26
    tl = t_local.set_index("partition").loc[PARTS]
    lm = tl.local_mean_client_mean
    ax.bar(x - w, lm, w, color="#DDDDDD", edgecolor="k", lw=0.3, label="Local only (mean client)")
    ax.errorbar(x - w, lm, yerr=[lm - tl.local_worst_client_mean, tl.local_best_client_mean - lm], fmt="none",
                ecolor="k", elinewidth=0.6, capsize=1.5)
    for i, s in enumerate(("fedavg", "fedadam")):
        ax.bar(x + i * w, tl[f"{s}_mean"], w, yerr=clip_err(tl[f"{s}_mean"], tl[f"{s}_std"]), color=COLOR[s],
               edgecolor="k", lw=0.3, capsize=1.2, error_kw={"lw": 0.5}, label=LABEL[s])
    ax.axhline(cm, color="k", ls="--", lw=0.9, label="Centralized")
    ax.set_xticks(x, [PLABEL[p].replace(" (", "\n(") for p in PARTS])
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2)
    save(fig, figd / "fig12_local_only.png")

    # console summary
    print(f"Friedman chi2 = {stat:.3f}, p = {pval:.3g} (k = {k}, n = {n}); Nemenyi CD(0.05) = {cd:.3f}")
    print("Mean ranks (1 = best):", ", ".join(f"{s} {r:.2f}" for s, r in mean_ranks.items()))
    print("Significant Nemenyi pairs (p < 0.05):", ", ".join(f"{a}-{b} ({p:.3g})" for a, b, p in sig_pairs) or "none")
    print("Cliques:", [[order[t] for t in range(i, j + 1)] for i, j in cliques])
    print(wt[["strategy", "median_diff", "mean_diff", "n_better", "n_worse", "rank_biserial_r", "p_raw", "p_holm",
              "significant_holm_0.05"]].to_string(index=False))
    print(r3(bt[["strategy", "mean_diff", "ci95_low", "ci95_high", "ci_excludes_zero"]]).to_string(index=False))
    if sens is not None:
        print(ts[["family", "strategy", "baseline", "n_better", "n_worse", "mean_diff", "p_holm", "ci95_low",
                  "ci95_high"]].to_string(index=False))
    print(r3(rt[["centralized_reference", "strategy", "n_runs", "coef_jsd", "p_jsd", "coef_cv", "p_cv", "r2", "corr_jsd_cv"]]).to_string(index=False))


if __name__ == "__main__":
    main()
