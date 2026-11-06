"""Preprocessing rules on tiny synthetic data (CLAUDE.md steps 5, 7, 8, 9 + an end-to-end run)."""

import numpy as np
import pandas as pd
import pytest

from fedsdn.data.preprocess import (
    DatasetSpec, Params, apply_labels, cap_mask, dedup, grouped_split, minute_cut_split,
    parse_timestamps, preprocess_frame, rare_rule, split_all,
)

T0 = pd.Timestamp("2020-01-10 12:00:00", tz="UTC")


def flows(rows):
    """rows: (row_id, seconds after T0, resolution_s, family, f1, f2)."""
    df = pd.DataFrame(rows, columns=["row_id", "t", "ts_resolution_s", "family", "f1", "f2"])
    df["ts"] = T0 + pd.to_timedelta(df.pop("t"), unit="s")
    df["ts_resolution_s"] = df["ts_resolution_s"].astype(np.int8)
    df["family"] = df["family"].astype("string")
    df["src_ip"], df["dst_ip"] = "10.0.0.1", "10.0.0.2"
    return df


# --- step 5: dedup ------------------------------------------------------------------------------

def test_dedup_drops_conflicting_groups_entirely_and_keeps_earliest_duplicate():
    df = flows([
        (0, 0, 60, "DoS", 1.0, 1.0),     # conflict group (1,1): DoS, DDoS, DoS -> all 3 dropped
        (1, 5, 60, "DDoS", 1.0, 1.0),
        (2, 9, 60, "DoS", 1.0, 1.0),
        (3, 50, 1, "DoS", 2.0, 2.0),     # duplicate group (2,2): keep earliest ts = row 4
        (4, 10, 1, "DoS", 2.0, 2.0),
        (5, 70, 1, "DoS", 2.0, 2.0),
        (6, 30, 1, "Probe", 3.0, 3.0),   # ts tie -> keep lower row_id = 6
        (7, 30, 1, "Probe", 3.0, 3.0),
        (8, 0, 1, "Normal", 4.0, 4.0),   # unique
        (9, 0, 1, "Normal", 4.0, 5.0),   # differs in one feature -> unique
    ])
    out, meta = dedup(df, ["f1", "f2"])
    assert out["row_id"].tolist() == [4, 6, 8, 9]
    assert meta["rows_dropped_conflicting"] == 3
    assert meta["conflicting_groups"] == 1
    assert meta["rows_dropped_conflicting_by_family"] == {"DDoS": 1, "DoS": 2}
    assert meta["rows_dropped_duplicate"] == 3
    assert meta["rows_after_dedup"] == 4


# --- step 7: rare rule --------------------------------------------------------------------------

def test_rare_rule_excludes_merged_class_below_threshold():
    counts = {"Normal": 1000, "DoS": 600, "BFA": 500, "Web": 300, "U2R": 100}
    d = rare_rule(counts, "Normal", 500, "Rare")
    assert d["rare_families"] == ["U2R", "Web"]          # 500 itself is not rare
    assert d["rare_total"] == 400 and d["rare_excluded_from_multi"]
    assert d["class_index"] == {"Normal": 0, "BFA": 1, "DoS": 2}

    df = pd.DataFrame({"family": pd.Series(["Normal", "DoS", "Web", "U2R", "BFA"], dtype="string")})
    lab = apply_labels(df, d, "Normal")
    assert lab["y_multi"].tolist() == [0, 2, -1, -1, 1]
    assert lab["y_bin"].tolist() == [0, 1, 1, 1, 1]      # excluded rows stay attacks in binary
    assert lab["cls"].tolist() == ["Normal", "DoS", "Rare", "Rare", "BFA"]


def test_rare_rule_keeps_merged_class_at_threshold():
    d = rare_rule({"Normal": 1000, "DoS": 600, "Web": 300, "U2R": 200}, "Normal", 500, "Rare")
    assert not d["rare_excluded_from_multi"]
    assert d["class_index"] == {"Normal": 0, "DoS": 1, "Rare": 2}


def test_rare_rule_no_rare_families():
    d = rare_rule({"Normal": 1000, "DoS": 600}, "Normal", 500, "Rare")
    assert d["rare_families"] == [] and not d["rare_excluded_from_multi"]
    assert d["class_index"] == {"Normal": 0, "DoS": 1}


# --- step 8: class cap --------------------------------------------------------------------------

def test_cap_thins_uniformly_in_time_order():
    # class A: 10 flows, row_id order deliberately differs from time order; ts tie at t=4.
    times = [9, 8, 7, 6, 5, 4, 4, 2, 1, 0]
    rows = [(i, t, 1, "A", float(i), 0.0) for i, t in enumerate(times)]
    rows += [(10 + i, i, 1, "B", 100.0 + i, 0.0) for i in range(3)]   # under cap: untouched
    df = flows(rows)
    df["cls"] = df["family"]
    capped = cap_mask(df, cap=4)
    # time order (ts, row_id): rows 9,8,7,5,6,4,3,2,1,0 ; keep positions floor(i*10/4) = 0,2,5,7
    kept_a = df.loc[(df["cls"] == "A") & ~capped, "row_id"].tolist()
    assert sorted(kept_a) == sorted([9, 7, 4, 2])
    assert not capped[df["cls"] == "B"].any()
    assert capped.sum() == 6


def test_cap_noop_at_exact_cap():
    df = flows([(i, i, 1, "A", float(i), 0.0) for i in range(5)])
    df["cls"] = df["family"]
    assert not cap_mask(df, cap=5).any()


# --- step 9: per-family split between minutes ------------------------------------------------

def minute_flows(counts, family="A", res=1, start_id=0, pairs=None):
    """counts[m] flows in minute m, spread over seconds 0..59 of that minute."""
    rows, rid = [], start_id
    for m, c in enumerate(counts):
        for k in range(c):
            rows.append((rid, 60 * m + (k * 59) // max(c - 1, 1), res, family, 0.0, 0.0))
            rid += 1
    df = flows(rows)
    if pairs is not None:
        df["src_ip"] = pairs
    df["cls"] = df["family"]
    return df


P = Params(test_target=0.2, test_share_min=0.10, test_share_max=0.35, min_test=0)


def test_minute_cut_closest_to_target_and_never_splits_a_minute():
    df = minute_flows([10] * 10)                      # 100 flows, 10 per minute, 1 s resolution
    split, info = minute_cut_split(df, P, min_test=0)
    minute = (df["ts"] - T0).dt.total_seconds() // 60
    assert (split[minute <= 7] == "train").all() and (split[minute >= 8] == "test").all()
    assert (split == "test").sum() == 20
    assert info["last_train_minute"] == str(T0 + pd.Timedelta(minutes=7))
    assert info["first_test_minute"] == str(T0 + pd.Timedelta(minutes=8))
    assert (split.groupby(minute).nunique() == 1).all()


def test_minute_cut_flows_at_minute_resolution_follow_their_minute():
    df = minute_flows([30, 30, 20, 20], res=60)       # cut after minute 2 -> test 20/100
    split, _ = minute_cut_split(df, P, min_test=0)
    assert split.tolist() == ["train"] * 80 + ["test"] * 20


def test_minute_cut_share_bounds_inclusive():
    assert minute_cut_split(minute_flows([65, 35]), P, 0)[0].eq("test").sum() == 35   # 0.35 ok
    assert minute_cut_split(minute_flows([90, 10]), P, 0)[0].eq("test").sum() == 10   # 0.10 ok
    assert minute_cut_split(minute_flows([64, 36]), P, 0)[0] is None                   # 0.36
    assert minute_cut_split(minute_flows([91, 9]), P, 0)[0] is None                    # 0.09


def test_minute_cut_respects_min_test():
    df = minute_flows([10] * 10)                      # best cut has 20 test flows
    assert minute_cut_split(df, P, min_test=21)[0].eq("test").sum() == 30
    assert minute_cut_split(df, P, min_test=36)[0] is None


def test_grouped_fallback_when_no_minute_cut_is_acceptable():
    # minutes [5, 5, 5, 85]: cuts give test shares 0.95 / 0.90 / 0.85 -> no acceptable cut.
    pairs = ["a"] * 60 + ["b"] * 22 + ["c"] * 18
    df = minute_flows([5, 5, 5, 85], pairs=pairs)
    assert minute_cut_split(df, P, 0)[0] is None
    split, info = grouped_split(df, P, min_test=0)
    # acceptable subset sums in [10, 35]: 18 (c), 22 (b); tie at distance 2 -> smaller (18)
    assert info["test_pairs"] == ["10.0.0.2|c"]
    assert (split == "test").sum() == 18
    assert (split.groupby(df["src_ip"]).nunique() == 1).all()

    s_all, per = split_all(df, pd.Series(False, index=df.index), [], P)
    assert per["A"]["method"] == "grouped_src_dst_pair" and per["A"]["test"] == 18
    assert per["A"]["minutes_shared"] >= 1            # grouped split may share minutes


def test_grouped_split_combines_pairs_and_treats_pairs_as_unordered():
    df = minute_flows([100])                          # one minute -> minute cut impossible
    df["src_ip"] = ["a"] * 70 + ["b"] * 12 + ["c"] * 10 + ["d"] * 8
    df.loc[df.index[:5], ["src_ip", "dst_ip"]] = ["10.0.0.2", "a"]    # reverse direction of a
    split, info = grouped_split(df, P, min_test=0)
    assert info["test_pairs"] == ["10.0.0.2|b", "10.0.0.2|d"]   # 12 + 8 = 20 exactly
    assert (split.iloc[:70] == "train").all()         # a (both directions) stays together


def test_relaxed_minute_cut_closest_to_range_up_to_50pct():
    def relaxed(counts, min_test=0):
        sp, _ = minute_cut_split(minute_flows(counts), P, min_test, relaxed=True)
        return None if sp is None else int((sp == "test").sum())
    assert minute_cut_split(minute_flows([57, 43]), P, 0)[0] is None    # strict: 0.43 too big
    assert relaxed([57, 43]) == 43
    assert relaxed([50, 8, 42]) == 42           # 0.50 vs 0.42 -> 0.42 is closer to 0.35
    assert relaxed([52, 40, 8]) == 8            # 0.48 (0.13 above) vs 0.08 (0.02 below)
    assert relaxed([50, 50]) == 50              # 0.50 allowed
    assert relaxed([49, 51]) is None            # 0.51 not allowed
    assert relaxed([52, 40, 8], min_test=9) == 48    # 8 test flows < min_test


def test_ddos_like_family_takes_the_relaxed_cut():
    counts = [2, 256, 209, 28, 2, 1, 373, 626, 23]     # InSDN DDoS per-minute counts
    df = minute_flows(counts, family="DDoS")
    df["src_ip"] = [f"1.0.{i // 256}.{i % 256}" for i in range(len(df))]   # one pair per flow
    split, per = split_all(df, pd.Series(False, index=df.index), [], Params())
    v = per["DDoS"]
    assert v["method"] == "minute_cut_relaxed"
    assert v["test"] == 649 and v["test_share"] == round(649 / 1520, 4)
    assert v["minutes_shared"] == 0 and "pairs_per_flow" not in v


def test_relaxed_cut_preferred_over_grouped_split():
    pairs = ["a"] * 60 + ["b"] * 22 + ["c"] * 18      # grouped split would be possible
    df = minute_flows([60, 40], pairs=pairs)           # strict none, relaxed 0.40
    _, per = split_all(df, pd.Series(False, index=df.index), [], P)
    assert per["A"]["method"] == "minute_cut_relaxed" and per["A"]["test"] == 40


def test_grouped_split_records_pairs_per_flow_and_flags_degenerate():
    df = minute_flows([5, 5, 5, 85], pairs=[f"p{i}" for i in range(100)])   # 1 pair per flow
    _, per = split_all(df, pd.Series(False, index=df.index), [], P)
    assert per["A"]["method"] == "grouped_src_dst_pair"
    assert per["A"]["pairs_per_flow"] == 1.0 and per["A"]["degenerate_grouping"]
    assert per["A"]["test"] == 20

    df = minute_flows([5, 5, 5, 85], pairs=["a"] * 60 + ["b"] * 22 + ["c"] * 18)
    _, per = split_all(df, pd.Series(False, index=df.index), [], P)
    assert per["A"]["pairs_per_flow"] == 0.03 and not per["A"]["degenerate_grouping"]


def test_split_all_rare_families_no_min_test_unsplittable_to_train_and_capped():
    p = Params(test_target=0.2, test_share_min=0.10, test_share_max=0.35, min_test=100)
    big = minute_flows([100] * 8 + [50] * 4, family="Big")          # 1000 flows
    small = minute_flows([10] * 10, family="Small", start_id=5000)   # non-rare, 100 flows
    rare = minute_flows([3] * 6, family="Tiny", start_id=6000)       # rare, 18 flows
    lone = minute_flows([7], family="Lone", start_id=7000)           # rare, 1 minute, 1 pair
    # BOTNET-like rare family: cuts give 51% (> 50%) or 3% (< 10%, not allowed for rare in (b))
    bot = minute_flows([80, 79, 5], family="Bot", start_id=8000)
    lowrare = minute_flows([58, 42], family="Half", start_id=9000)    # rare, (b) at 42% is ok
    df = pd.concat([big, small, rare, lone, bot, lowrare], ignore_index=True)
    capped = pd.Series(False, index=df.index)
    capped.iloc[:5] = True
    split, per = split_all(df, capped, ["Bot", "Half", "Lone", "Tiny"], p)

    assert (split.iloc[:5] == "capped").all()
    assert set(split.iloc[5:]) <= {"train", "test"}
    assert per["Big"]["method"] == "minute_cut" and per["Big"]["test"] == 200
    assert per["Big"]["minutes_shared"] == 0
    assert per["Small"]["method"] == "unsplit_all_train" and per["Small"]["test"] == 0
    assert per["Tiny"]["method"] == "minute_cut" and per["Tiny"]["test"] == 3       # 1/6 closest
    assert per["Lone"]["method"] == "unsplit_all_train" and per["Lone"]["train"] == 7
    assert per["Bot"]["method"] == "unsplit_all_train" and per["Bot"]["train"] == 164
    assert per["Half"]["method"] == "minute_cut_relaxed" and per["Half"]["test"] == 42


# --- steps 2-10 end to end ----------------------------------------------------------------------

def test_parse_timestamps_both_formats():
    ts, res = parse_timestamps(pd.Series(["5/2/2020 13:58", "31/12/2019 02:25:57 PM", "bad"]),
                               DatasetSpec("x", ()).ts_formats)
    assert ts.iloc[0] == pd.Timestamp("2020-02-05 13:58", tz="UTC") and res.iloc[0] == 60
    assert ts.iloc[1] == pd.Timestamp("2019-12-31 14:25:57", tz="UTC") and res.iloc[1] == 1
    assert pd.isna(ts.iloc[2]) and res.iloc[2] == 0


def test_preprocess_frame_end_to_end():
    rng = np.random.default_rng(0)
    n = 300
    fam = np.array(["Normal "] * 150 + ["DoS"] * 120 + ["U2R"] * 30, dtype=object)
    minutes = np.concatenate([np.arange(150), np.arange(120), np.arange(30)]) * 3
    raw = pd.DataFrame({
        "Flow ID": [f"id{i}" for i in range(n)],
        " Src IP": [f"10.0.0.{i % 7}" for i in range(n)],
        "Src Port": rng.integers(1000, 2000, n),
        "Dst IP": "10.0.0.100",
        "Dst Port": rng.integers(1, 100, n),
        "Protocol": rng.choice([6, 17], n),
        "Timestamp": [(pd.Timestamp("2020-01-10 08:00") + pd.Timedelta(minutes=int(m)))
                      .strftime("%d/%m/%Y %H:%M").lstrip("0") for m in minutes],
        "Flow Duration": rng.random(n),
        "Const Col": 7.0,
        "Label": fam,
    })
    raw.loc[5, "Flow Duration"] = np.inf                              # step 3
    raw.loc[10, ["Dst Port", "Protocol", "Flow Duration"]] = raw.loc[11, ["Dst Port", "Protocol",
                                                                            "Flow Duration"]].values
    raw.loc[10, "Label"] = "DoS"                                      # conflict with row 11
    raw.insert(0, "row_id", np.arange(n, dtype=np.int64))
    out, meta = preprocess_frame(raw, DatasetSpec("x", ()),
                                 Params(rare_min=50, cap=100, min_test=10))
    assert meta["step3_invalid"]["rows_dropped_invalid"] == 1
    assert meta["step5_dedup"]["rows_dropped_conflicting"] == 2
    assert meta["rare_class_decision"]["rare_families"] == ["U2R"]
    assert (out.loc[out["family"] == "U2R", "y_multi"] == -1).all()
    assert meta["class_counts_after_cap"]["Normal"]["post_cap"] == 100
    assert {c for c in out.columns if c.startswith("f_")} == {
        "f_dst_port", "f_flow_duration", "f_protocol_6", "f_protocol_17"}
    assert [d["column"] for d in meta["features"]["dropped_columns"]] == ["Const Col"]
    assert set(out.loc[~out["capped_out"], "split"]) == {"train", "test"}
    assert (out.loc[out["capped_out"], "split"] == "capped").all()
    assert out["f_dst_port"].dtype == np.float32 and out["y_multi"].dtype == np.int16
    assert str(out["ts"].dt.tz) == "UTC"
