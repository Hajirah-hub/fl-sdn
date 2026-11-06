"""Dataset preprocessing: CLAUDE.md "Preprocessing rules", steps 1-11, applied in order.

Every step is a small pure function on a DataFrame so it can be unit-tested on synthetic data;
`preprocess_frame` chains steps 2-10 and `build` adds step 1 (loading) and the atomic writes.

Output: `<out_dir>/flows.parquet` (one row per flow surviving steps 1-5; capped-out and
multi-class-excluded rows are kept and marked) and `<out_dir>/meta.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Dataset spec and parameters
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    files: tuple[str, ...]
    readme_counts: dict[str, int] = field(default_factory=dict)
    label_col: str = "Label"
    ts_col: str = "Timestamp"
    src_ip_col: str = "Src IP"
    dst_ip_col: str = "Dst IP"
    # Step 4: identifiers removed from the feature set. `Dst Port` is deliberately NOT here.
    id_cols: tuple[str, ...] = ("Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp")
    protocol_col: str = "Protocol"
    benign_label: str = "Normal"
    # (strptime format, resolution in seconds), tried in order; anything else -> unparsed (NaT).
    ts_formats: tuple[tuple[str, int], ...] = (
        ("%d/%m/%Y %H:%M", 60),          # 5/2/2020 13:58          (24 h, minute resolution)
        ("%d/%m/%Y %I:%M:%S %p", 1),     # 31/12/2019 02:25:57 PM  (12 h, second resolution)
    )


INSDN = DatasetSpec(
    name="insdn",
    files=("Normal_data.csv", "OVS.csv", "metasploitable-2.csv"),
    readme_counts={"Normal_data.csv": 68_424, "OVS.csv": 136_743, "metasploitable-2.csv": 138_772},
)

SPECS = {"insdn": INSDN}


@dataclass(frozen=True)
class Params:
    rare_min: int = 500          # step 7
    rare_name: str = "Rare"
    cap: int = 20_000            # step 8
    test_target: float = 0.20    # step 9: test share aimed for
    test_share_min: float = 0.10 # acceptable test share range (minute cut and grouped fallback)
    test_share_max: float = 0.35
    min_test: int = 100          # minimum test flows (not applied to rare families)
    relaxed_share_max: float = 0.50       # relaxed minute cut: max test share
    degenerate_pairs_per_flow: float = 0.50  # grouped split flagged above this


# --------------------------------------------------------------------------------------------
# Step 1: load
# --------------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_raw(raw_dir: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    """Concatenate the CSVs in file-name order; `row_id` = global index in (file name, file row)."""
    frames, files_meta = [], {}
    for fname in sorted(spec.files):
        path = Path(raw_dir) / fname
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        expected = spec.readme_counts.get(fname)
        files_meta[fname] = {
            "sha256": sha256_file(path),
            "rows_readme": expected,
            "rows_observed": len(df),
            "difference": None if expected is None else len(df) - expected,
        }
        frames.append(df)
    cols = list(frames[0].columns)
    for fname, df in zip(sorted(spec.files), frames):
        if list(df.columns) != cols:
            raise ValueError(f"{fname}: column names/order differ from {sorted(spec.files)[0]}")
    raw = pd.concat(frames, ignore_index=True)
    raw.insert(0, "row_id", np.arange(len(raw), dtype=np.int64))
    return raw, {"files": files_meta, "rows_raw_total": len(raw)}


# --------------------------------------------------------------------------------------------
# Step 2: strip whitespace, parse timestamps
# --------------------------------------------------------------------------------------------


def parse_timestamps(ts_str: pd.Series, formats) -> tuple[pd.Series, pd.Series]:
    """Explicit-format parse. Returns (`ts` UTC, `ts_resolution_s` int8; 0 where unparsed)."""
    ts_str = ts_str.astype("string").str.strip()
    ts = pd.Series(pd.NaT, index=ts_str.index, dtype="datetime64[ns, UTC]")
    res = pd.Series(0, index=ts_str.index, dtype=np.int8)
    for fmt, resolution in formats:
        todo = ts.isna() & ts_str.notna()
        if not todo.any():
            break
        parsed = pd.to_datetime(ts_str[todo], format=fmt, errors="coerce", utc=True)
        ok = parsed.index[parsed.notna()]
        ts.loc[ok] = parsed.loc[ok]
        res.loc[ok] = resolution
    return ts, res


def strip_and_parse(raw: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("string").str.strip()
    df["ts"], df["ts_resolution_s"] = parse_timestamps(df[spec.ts_col], spec.ts_formats)
    meta = {
        "rows_by_resolution_s": {str(k): int(v) for k, v in
                                 df["ts_resolution_s"].value_counts().sort_index().items()},
        "ts_unparsed": int(df["ts"].isna().sum()),
        "ts_min": str(df["ts"].min()),
        "ts_max": str(df["ts"].max()),
        "ts_note": "wall-clock times from the CSV, interpreted as UTC (no zone in source)",
    }
    return df, meta


# --------------------------------------------------------------------------------------------
# Steps 3-4: invalid rows, feature / metadata separation
# --------------------------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    excluded = set(spec.id_cols) | {spec.label_col, "row_id", "ts", "ts_resolution_s"}
    return [c for c in df.columns if c not in excluded]


def drop_invalid(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, dict]:
    """Step 3: drop rows with missing or infinite values (features, label, ts, IPs)."""
    df = df.copy()
    feats = feature_columns(df, spec)
    coerced = 0
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            before = df[c].isna().sum()
            df[c] = pd.to_numeric(df[c], errors="coerce")
            coerced += int(df[c].isna().sum() - before)
        df[c] = df[c].astype(np.float64)
    x = df[feats].to_numpy()
    nan_feat = np.isnan(x).any(axis=1)
    inf_feat = np.isinf(x).any(axis=1)
    meta_cols = [spec.label_col, spec.src_ip_col, spec.dst_ip_col, "ts"]
    nan_meta = df[meta_cols].isna().any(axis=1).to_numpy()
    bad = nan_feat | inf_feat | nan_meta
    meta = {
        "rows_with_nan_feature": int(nan_feat.sum()),
        "rows_with_inf_feature": int(inf_feat.sum()),
        "rows_with_missing_label_ip_or_ts": int(nan_meta.sum()),
        "non_numeric_feature_cells_coerced": coerced,
        "rows_dropped_invalid": int(bad.sum()),
    }
    return df.loc[~bad].reset_index(drop=True), meta


def separate(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, list[str]]:
    """Step 4: keep metadata (ts, src_ip, dst_ip, family) + feature columns; drop identifiers."""
    feats = feature_columns(df, spec)
    out = pd.DataFrame({
        "row_id": df["row_id"].astype(np.int64),
        "ts": df["ts"],
        "ts_resolution_s": df["ts_resolution_s"].astype(np.int8),
        "src_ip": df[spec.src_ip_col].astype("string"),
        "dst_ip": df[spec.dst_ip_col].astype("string"),
        "family": df[spec.label_col].astype("string"),
    })
    out = pd.concat([out, df[feats]], axis=1)
    return out, feats


# --------------------------------------------------------------------------------------------
# Step 5: deduplication
# --------------------------------------------------------------------------------------------


def dedup(df: pd.DataFrame, feats: list[str], label: str = "family") -> tuple[pd.DataFrame, dict]:
    """Drop every feature-identical group with >1 distinct label entirely, then keep the earliest
    (ts, then row_id) of each exact feature+label duplicate group."""
    gid = df.groupby(feats, sort=False, dropna=False).ngroup()
    n_labels = df[label].groupby(gid).transform("nunique")
    conflict = (n_labels > 1).to_numpy()
    kept = df.loc[~conflict].sort_values(["ts", "row_id"], kind="stable")
    dup = kept.duplicated(subset=feats + [label], keep="first").to_numpy()
    out = kept.loc[~dup].sort_values("row_id", kind="stable").reset_index(drop=True)
    meta = {
        "rows_dropped_conflicting": int(conflict.sum()),
        "conflicting_groups": int(gid[conflict].nunique()),
        "rows_dropped_conflicting_by_family": _counts(df.loc[conflict, label]),
        "rows_dropped_duplicate": int(dup.sum()),
        "rows_dropped_duplicate_by_family": _counts(kept.loc[dup, label]),
        "rows_after_dedup": len(out),
    }
    return out, meta


# --------------------------------------------------------------------------------------------
# Steps 6-7: labels, rare-class rule
# --------------------------------------------------------------------------------------------


def rare_rule(family_counts: dict[str, int], benign: str, rare_min: int, rare_name: str) -> dict:
    """Families with < rare_min flows merge into `rare_name`; if the merged class still has
    < rare_min flows it is excluded from the multi-class task (y_multi = -1).

    Class index: benign = 0, then the remaining classes alphabetically, then `rare_name` last."""
    rare = sorted(f for f, n in family_counts.items() if n < rare_min)
    if benign in rare:
        raise ValueError(f"benign class {benign!r} has < {rare_min} flows")
    rare_total = sum(family_counts[f] for f in rare)
    excluded = bool(rare) and rare_total < rare_min
    classes = [benign] + sorted(f for f in family_counts if f != benign and f not in rare)
    if rare and not excluded:
        classes.append(rare_name)
    return {
        "rare_min": rare_min,
        "rare_families": rare,
        "rare_total": rare_total,
        "rare_excluded_from_multi": excluded,
        "class_of_family": {f: (rare_name if f in rare else f) for f in sorted(family_counts)},
        "class_index": {c: i for i, c in enumerate(classes)},
    }


def apply_labels(df: pd.DataFrame, decision: dict, benign: str) -> pd.DataFrame:
    df = df.copy()
    df["cls"] = df["family"].map(decision["class_of_family"]).astype("string")
    df["y_multi"] = df["cls"].map(decision["class_index"]).fillna(-1).astype(np.int16)
    df["y_bin"] = (df["family"] != benign).astype(np.int8)
    return df


# --------------------------------------------------------------------------------------------
# Step 8: class cap
# --------------------------------------------------------------------------------------------


def cap_mask(df: pd.DataFrame, cap: int, group: str = "cls") -> pd.Series:
    """True for rows removed by the cap. Classes above `cap` keep `cap` evenly spaced positions
    (floor(i * n / cap), i = 0..cap-1) in (ts, row_id) order."""
    capped = pd.Series(False, index=df.index)
    for _, g in df.groupby(group, sort=True):
        n = len(g)
        if n <= cap:
            continue
        order = g.sort_values(["ts", "row_id"], kind="stable").index
        drop = np.ones(n, dtype=bool)
        drop[(np.arange(cap, dtype=np.int64) * n) // cap] = False
        capped.loc[order[drop]] = True
    return capped


# --------------------------------------------------------------------------------------------
# Step 9: per-family split between minutes (relaxed cut, grouped fallback, then all-train)
# --------------------------------------------------------------------------------------------


def _share_ok(n_test, n: int, p: "Params", min_test: int):
    """Acceptable test sizes: >= max(min_test, 1) flows and share within [share_min, share_max]."""
    share = np.asarray(n_test) / n
    return (np.asarray(n_test) >= max(min_test, 1)) & (share >= p.test_share_min) & (share <= p.test_share_max)


def _dist(n_test, n: int, p: "Params") -> np.ndarray:
    """|test share - target|, rounded so exact ties are not decided by float noise."""
    return np.round(np.abs(np.asarray(n_test) / n - p.test_target), 12)


def _dist_to_range(n_test, n: int, p: "Params") -> np.ndarray:
    """Distance of the test share from [test_share_min, test_share_max] (0 inside), rounded."""
    share = np.asarray(n_test) / n
    return np.round(np.maximum.reduce([p.test_share_min - share, share - p.test_share_max,
                                       np.zeros_like(share)]), 12)


def minute_cut_split(g: pd.DataFrame, p: "Params", min_test: int, relaxed: bool = False,
                     relaxed_min_share: float = 0.0) -> tuple[pd.Series | None, dict]:
    """Floor timestamps to the minute and cut between two consecutive minutes (train = minutes
    <= the cut).

    Strict (rule a): among acceptable cuts (`_share_ok`), the one whose test share is closest to
    `test_target`. Relaxed (rule b): among cuts with >= max(min_test, 1) test flows and test share
    in [`relaxed_min_share`, `relaxed_share_max`], the one closest to [test_share_min,
    test_share_max]. Ties -> earliest cut. Returns (None, {}) if no cut qualifies."""
    n = len(g)
    minute = g["ts"].dt.floor("min")
    counts = minute.value_counts().sort_index()
    n_test = n - counts.cumsum().to_numpy()          # test size if the cut is after minute i
    if relaxed:
        share = n_test / n
        ok = (n_test >= max(min_test, 1)) & (share >= relaxed_min_share) & (share <= p.relaxed_share_max)
        dist = _dist_to_range(n_test, n, p)
    else:
        ok = _share_ok(n_test, n, p, min_test)
        dist = _dist(n_test, n, p)
    if not ok.any():
        return None, {}
    i = int(np.argmin(np.where(ok, dist, np.inf)))
    last_train = counts.index[i]
    split = pd.Series(np.where(minute <= last_train, "train", "test"), index=g.index, dtype="string")
    return split, {"last_train_minute": str(last_train), "first_test_minute": str(counts.index[i + 1])}


def grouped_split(g: pd.DataFrame, p: "Params", min_test: int) -> tuple[pd.Series | None, dict]:
    """Fallback: whole unordered src-dst address pairs go to test. Exact subset-sum over pair
    sizes picks the acceptable test size (`_share_ok`) closest to `test_target` (ties -> the
    smaller size); the subset is reconstructed deterministically over pairs sorted by name.
    Returns (None, {}) if no acceptable subset exists."""
    n = len(g)
    pair = pd.Series(["|".join(sorted(x)) for x in zip(g["src_ip"], g["dst_ip"])], index=g.index)
    sizes = pair.value_counts().sort_index()
    if len(sizes) < 2:
        return None, {}
    full = (1 << (n + 1)) - 1
    reach = [1]                                      # bit t set <=> test size t reachable
    for s in sizes.to_numpy():
        reach.append((reach[-1] | (reach[-1] << int(s))) & full)
    sums = np.arange(n + 1)
    reachable = np.array([(reach[-1] >> int(t)) & 1 for t in sums], dtype=bool)
    ok = reachable & _share_ok(sums, n, p, min_test)
    if not ok.any():
        return None, {}
    t = int(sums[np.argmin(np.where(ok, _dist(sums, n, p), np.inf))])
    test_pairs = []
    for i in range(len(sizes), 0, -1):
        if not (reach[i - 1] >> t) & 1:              # t not reachable without pair i-1 -> take it
            test_pairs.append(sizes.index[i - 1])
            t -= int(sizes.iloc[i - 1])
    assert t == 0
    split = pd.Series(np.where(pair.isin(test_pairs), "test", "train"), index=g.index, dtype="string")
    return split, {"n_pairs": len(sizes), "test_pairs": sorted(test_pairs)}


def pairs_per_flow(g: pd.DataFrame) -> float:
    """Distinct unordered src-dst pairs / flows (1.0 = every flow its own pair)."""
    pairs = {"|".join(sorted(x)) for x in zip(g["src_ip"], g["dst_ip"])}
    return len(pairs) / len(g)


def split_all(df: pd.DataFrame, capped: pd.Series, rare_families: list[str],
              p: "Params") -> tuple[pd.Series, dict]:
    """Per-family split of the non-capped rows; capped rows get split = "capped".

    Order per family (first that succeeds): (a) strict minute cut, (b) relaxed minute cut,
    (c) grouped src-dst split, (d) whole family to train (`unsplit_all_train`).
    Rare families have no `min_test`, and for them rule (b) also needs a test share
    >= `test_share_min` (so it can only pick a share in (test_share_max, relaxed_share_max]). Grouped families record pairs-per-flow and are flagged
    `degenerate_grouping` above `degenerate_pairs_per_flow`."""
    split = pd.Series("capped", index=df.index, dtype="string")
    per_family = {}
    for fam, g in df.loc[~capped].groupby("family", sort=True):
        fam = str(fam)
        rare = fam in rare_families
        min_test = 0 if rare else p.min_test
        s, info = minute_cut_split(g, p, min_test)
        method = "minute_cut"
        if s is None:
            s, info = minute_cut_split(g, p, min_test, relaxed=True,
                                       relaxed_min_share=p.test_share_min if rare else 0.0)
            method = "minute_cut_relaxed"
        if s is None:
            s, info = grouped_split(g, p, min_test)
            method = "grouped_src_dst_pair"
            if s is not None:
                ppf = pairs_per_flow(g)
                info = {**info, "pairs_per_flow": round(ppf, 4),
                        "degenerate_grouping": ppf > p.degenerate_pairs_per_flow}
        if s is None:
            s, info = pd.Series("train", index=g.index, dtype="string"), {}
            method = "unsplit_all_train"
        split.loc[s.index] = s
        minute = g["ts"].dt.floor("min")
        m_train, m_test = set(minute[s == "train"]), set(minute[s == "test"])
        n_test = int((s == "test").sum())
        per_family[fam] = {
            "cls": str(g["cls"].iloc[0]),
            "rare_family": rare,
            "min_test": min_test,
            "method": method,
            "n": len(g),
            "train": len(g) - n_test,
            "test": n_test,
            "test_share": round(n_test / len(g), 4),
            "minutes_train": len(m_train),
            "minutes_test": len(m_test),
            "minutes_shared": len(m_train & m_test),
            **info,
        }
    return split, per_family


# --------------------------------------------------------------------------------------------
# Step 10: constant columns, protocol one-hot
# --------------------------------------------------------------------------------------------


def feature_name(col: str) -> str:
    return "f_" + re.sub(r"[^0-9a-z]+", "_", col.lower()).strip("_")


def build_features(df: pd.DataFrame, feats: list[str], train_pool: pd.Series,
                   protocol_col: str) -> tuple[pd.DataFrame, dict]:
    """Drop features constant over the training pool; one-hot `protocol_col` with the categories
    seen in the training pool. Returns float32 `f_*` columns and the feature metadata."""
    pool = df.loc[train_pool, feats]
    nunique = pool.nunique(dropna=False)
    dropped = [
        {"column": c, "reason": "constant over training pool", "value": float(pool[c].iloc[0])}
        for c in feats if nunique[c] <= 1
    ]
    dropped_names = {d["column"] for d in dropped}
    numeric = [c for c in feats if c not in dropped_names and c != protocol_col]

    out, sources = {}, {}
    for c in numeric:
        name = feature_name(c)
        if name in out:
            raise ValueError(f"feature name collision: {c!r} -> {name}")
        out[name], sources[name] = df[c].astype(np.float32).to_numpy(), c
    categories = []
    if protocol_col in feats and protocol_col not in dropped_names:
        categories = sorted(pool[protocol_col].unique().tolist())
        for v in categories:
            name = f"{feature_name(protocol_col)}_{int(v)}"
            out[name] = (df[protocol_col] == v).astype(np.float32).to_numpy()
            sources[name] = f"{protocol_col} == {int(v)} (one-hot)"
    meta = {
        "n_features": len(out),
        "features": [{"name": k, "source": v} for k, v in sources.items()],
        "dropped_columns": dropped,
        "protocol_categories": [int(v) for v in categories],
    }
    return pd.DataFrame(out, index=df.index), meta


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def _counts(s: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}


def preprocess_frame(raw: pd.DataFrame, spec: DatasetSpec, p: Params = Params()) -> tuple[pd.DataFrame, dict]:
    """Steps 2-10 on a raw frame that already has `row_id` (step 1). Step 11: no scaling here."""
    meta: dict = {"params": asdict(p)}
    df, meta["step2_timestamps"] = strip_and_parse(raw, spec)
    df, meta["step3_invalid"] = drop_invalid(df, spec)
    df, feats = separate(df, spec)
    meta["step4_identifiers_removed"] = list(spec.id_cols)
    df, meta["step5_dedup"] = dedup(df, feats)
    meta["class_counts_after_dedup"] = _counts(df["family"])

    decision = rare_rule(meta["class_counts_after_dedup"], spec.benign_label, p.rare_min, p.rare_name)
    df = apply_labels(df, decision, spec.benign_label)
    meta["rare_class_decision"] = decision
    meta["class_index"] = decision["class_index"]
    meta["class_counts_after_rare_rule"] = _counts(df["cls"])

    capped = cap_mask(df, p.cap)
    df["capped_out"] = capped.to_numpy()
    meta["class_counts_after_cap"] = {
        c: {"pre_cap": int(n), "post_cap": int((~capped[df["cls"] == c]).sum())}
        for c, n in df["cls"].value_counts().sort_index().items()
    }

    df["split"], per_family = split_all(df, capped, decision["rare_families"], p)
    meta["split_per_family"] = per_family
    meta["split_method_per_family"] = {f: v["method"] for f, v in per_family.items()}
    meta["unsplit_families"] = [f for f, v in per_family.items() if v["method"] == "unsplit_all_train"]
    meta["degenerate_grouping_flags"] = [f for f, v in per_family.items() if v.get("degenerate_grouping")]
    meta["class_counts_per_split"] = {
        s: _counts(df.loc[df["split"] == s, "cls"]) for s in ("train", "test", "capped")
    }
    meta["binary_counts_per_split"] = {
        s: _counts(df.loc[df["split"] == s, "y_bin"]) for s in ("train", "test", "capped")
    }

    train_pool = (~capped) & (df["split"] == "train")
    fx, meta["features"] = build_features(df, feats, train_pool.to_numpy(), spec.protocol_col)

    out = pd.concat([
        df[["row_id", "ts", "ts_resolution_s", "src_ip", "dst_ip", "family",
            "y_multi", "y_bin", "capped_out", "split"]],
        fx,
    ], axis=1)
    out["family"] = out["family"].astype("string")
    meta["rows_out"] = len(out)
    fcols = [c for c in out.columns if c.startswith("f_")]
    meta["informational"] = {
        "final_feature_space_duplicate_rows_among_live": int(
            out.loc[~out["capped_out"], fcols].duplicated().sum()),
    }
    return out, meta


# --------------------------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------------------------


def atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def build(raw_dir: Path, out_dir: Path, spec: DatasetSpec, p: Params = Params()) -> dict:
    raw, load_meta = load_raw(raw_dir, spec)
    flows, meta = preprocess_frame(raw, spec, p)
    meta = {"dataset": spec.name, "step1_load": load_meta, **meta}
    out_dir = Path(out_dir)
    atomic_write(out_dir / "flows.parquet", lambda t: flows.to_parquet(t, index=False))
    meta["flows_sha256"] = sha256_file(out_dir / "flows.parquet")
    atomic_write(out_dir / "meta.json",
                  lambda t: Path(t).write_text(json.dumps(meta, indent=2), encoding="utf-8"))
    return meta


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def format_report(meta: dict) -> str:
    L: list[str] = []
    add = L.append
    add(f"=== {meta.get('dataset', '?')} preprocessing report ===")
    if "step1_load" in meta:
        add("\nRaw files (README vs observed):")
        for f, v in meta["step1_load"]["files"].items():
            add(f"  {f:<22} readme={v['rows_readme']!s:>8} observed={v['rows_observed']:>8} "
                f"diff={v['difference']!s:>6}")
    t = meta["step2_timestamps"]
    add(f"\nTimestamps: by resolution {t['rows_by_resolution_s']}, unparsed {t['ts_unparsed']}, "
        f"range {t['ts_min']} .. {t['ts_max']}")
    add(f"Invalid rows dropped: {meta['step3_invalid']['rows_dropped_invalid']}")
    d = meta["step5_dedup"]
    add(f"Dedup: conflicting rows dropped {d['rows_dropped_conflicting']} "
        f"({d['conflicting_groups']} groups) {d['rows_dropped_conflicting_by_family']}")
    add(f"       duplicate rows dropped  {d['rows_dropped_duplicate']} {d['rows_dropped_duplicate_by_family']}")
    add(f"       rows after dedup {d['rows_after_dedup']}")

    add("\nClass counts after dedup (family):")
    for c, n in meta["class_counts_after_dedup"].items():
        add(f"  {c:<12} {n:>8}")
    r = meta["rare_class_decision"]
    add(f"\nRare rule (< {r['rare_min']}): families {r['rare_families']} -> total {r['rare_total']}, "
        f"excluded from multi-class: {r['rare_excluded_from_multi']}")
    add(f"Class index: {meta['class_index']}")
    add("Class counts after rare rule:")
    for c, n in meta["class_counts_after_rare_rule"].items():
        add(f"  {c:<12} {n:>8}  y_multi={meta['class_index'].get(c, -1)}")

    add("\nClass cap (pre -> post):")
    for c, v in meta["class_counts_after_cap"].items():
        add(f"  {c:<12} {v['pre_cap']:>8} -> {v['post_cap']:>8}")

    add("\nSplit per family (after cap; minute cut, else relaxed cut, else grouped src-dst, else all train):")
    add(f"  {'family':<11} {'class':<7} {'n':>6} {'train':>6} {'test':>6} {'share':>6} "
        f"{'min_tr':>6} {'min_te':>6} {'shared':>6}  method / cut")
    for fam, v in meta["split_per_family"].items():
        if v["method"] in ("minute_cut", "minute_cut_relaxed"):
            where = f"{v['method']}  {v['last_train_minute'][:16]} | {v['first_test_minute'][:16]}"
        elif v["method"] == "grouped_src_dst_pair":
            tp = v["test_pairs"]
            shown = ", ".join(tp[:2]) + (f", ... (+{len(tp) - 2})" if len(tp) > 2 else "")
            where = (f"grouped  {len(tp)} of {v['n_pairs']} pairs in test: {shown}; "
                     f"pairs/flow {v['pairs_per_flow']:.3f}"
                     + ("  <-- DEGENERATE GROUPING" if v["degenerate_grouping"] else ""))
        else:
            where = "UNSPLIT (all train)"
        add(f"  {fam:<11} {v['cls']:<7} {v['n']:>6} {v['train']:>6} {v['test']:>6} "
            f"{v['test_share']:>6.3f} {v['minutes_train']:>6} {v['minutes_test']:>6} "
            f"{v['minutes_shared']:>6}  {where}")
    add("Class counts per split:")
    cps = meta["class_counts_per_split"]
    for c in meta["class_counts_after_rare_rule"]:
        tr, te = cps["train"].get(c, 0), cps["test"].get(c, 0)
        add(f"  {c:<12} train {tr:>6}  test {te:>6}  test share {te / max(tr + te, 1):.3f}")
    tr, te = sum(cps["train"].values()), sum(cps["test"].values())
    add(f"  {'TOTAL':<12} train {tr:>6}  test {te:>6}  test share {te / max(tr + te, 1):.3f}")
    add(f"Binary counts per split (0 benign / 1 attack): {meta['binary_counts_per_split']}")
    add(f"Unsplit families: {meta['unsplit_families'] or 'none'}")
    add(f"Degenerate grouping (pairs/flow > {meta['params']['degenerate_pairs_per_flow']}): "
        f"{meta['degenerate_grouping_flags'] or 'none'}")

    f = meta["features"]
    add(f"\nFeatures: {f['n_features']} (protocol one-hot categories {f['protocol_categories']})")
    add(f"Dropped columns ({len(f['dropped_columns'])}, constant over training pool):")
    for dcol in f["dropped_columns"]:
        add(f"  {dcol['column']:<20} value={dcol['value']:g}")
    add(f"Identifiers removed: {meta['step4_identifiers_removed']}")
    add(f"Info: duplicate feature vectors among live rows in final float32 feature space: "
        f"{meta['informational']['final_feature_space_duplicate_rows_among_live']}")
    return "\n".join(L)
