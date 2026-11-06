# Experiment write-up: federated aggregation strategies on InSDN (study v2)

This document records what was actually implemented and run, so the implementation, results and
discussion sections of the paper can be written from it. Every number comes from the code, the git
history, `CLAUDE.md`, `configs/tuning/insdn.yaml`, `data/processed/*/meta.json`, the partition
sidecars, `results/<suite>/summary.csv`, `results/paper_tables/` or `results/paper_figures/`. Where
something is not recorded, this document says so. Numbers that were computed only for this
write-up (from stored outputs, not part of the pipeline) are marked **[computed for this
write-up]**.

Section III of `docs/paper.pdf` is called "the paper" or "the plan" below.

**Version.** This is the **v2** study (tuned hyperparameters, SCAFFOLD, 5 seeds, sweeps, local-only
baseline, poisoning on two protocols, full statistics). The earlier untuned 3-seed study (v1:
7 strategies, client lr 0.05) is archived unchanged in `results/archive_v1/` and is described by
the version of this file at commit `6de8a9d`. v1 numbers are not used anywhere below except in
Section 15 (Flower validation), which was run in v1.

---

## 1. Overview

### Research questions (from the paper)

- **RQ1:** How much does controller-level heterogeneity (label skew, quantity skew, attack-type
  exclusivity) degrade FedAvg-based threat detection compared with centralized and IID-federated
  training?
- **RQ2:** Which heterogeneity-aware or adaptive aggregation strategies recover that loss, and at
  what cost in rounds and bytes?
- **RQ3:** How do Byzantine-robust rules (Krum, coordinate-wise median, trimmed mean) behave with
  heterogeneous honest controllers, and when some controllers send poisoned updates?

### What was run

The full factorial design of the paper (4 datasets, 2 tasks, 10 partitions, 8 strategies, 5–10
seeds, K and E sweeps, about 8,000 runs) was not run. A **reduced study** was run instead, in this
order, every later step using the hyperparameters frozen in step 1:

1. **Tuning** (paper Section III-H): 12 random-search trials per strategy and for the centralized
   baseline, random protocol, R = 30, tuning seed 99 on its own partitions (P1 α = 0.5, P3 m = 1,
   P4), scored on pooled **validation** macro-F1 only. Frozen in `configs/tuning/insdn.yaml`.
2. **SCAFFOLD** implemented (8 strategies in total).
3. **Main grid, random protocol:** 8 strategies × P0 / P1 α = 0.5 / P3 m = 1 / P4 × seeds 0–4,
   R = 100; centralized baseline 100 epochs.
4. **Same grid on the chronological protocol,** R = 30, plus poisoning on chronological P4
   (FedAvg, Krum, Median, Trimmed Mean; f = 2 of 8; sign flip and targeted relabeling).
5. **Heterogeneity sweep,** random, R = 100: FedAvg and FedAdam on P1 α ∈ {0.1, 1, 10} and
   P2 β = 0.5.
6. **Local-only baseline** (random, 100 epochs per client, global test set) and **poisoning on
   random P3 m = 1** (R = 100), recording whether a malicious controller is the sole holder of an
   attack family.
7. **Learning-rate checks** (requested after tuning): a boundary check on the tuning seed (lr 0.4 /
   0.6, validation only) and a sensitivity grid (FedAvg / FedProx at lr 0.6 on the random grid).
8. **Statistics** on the random protocol, R = 100 (20 partition × seed blocks): Friedman + Nemenyi,
   Wilcoxon vs FedAvg with Holm, bootstrap CIs, and a regression of the gap to centralized on
   realized JSD and CV.

Fixed throughout: InSDN, multi-class (Normal, BFA, DDoS, DoS, Probe), K = 8 controllers, E = 1,
batch 256, plain SGD, MLP, f̂ = 2 for the robust rules. Simulation in **plain PyTorch** (all clients
of a round run one after another in one process), validated against Flower in Section 15.

Results used below: 300 tuning runs, 165 random-protocol runs (R = 100), 205 chronological runs
(R = 30, incl. 40 poisoning), 40 sweep runs, 20 local-only runs, 40 random-protocol poisoning runs,
24 boundary-check runs and 40 sensitivity runs. The headline metric is **validation-selected test
macro-F1**: the test score at the round with the best pooled validation macro-F1.

### Main findings

1. **Tuning mattered, and the optimum sits at the top of the search range.** All plain-SGD rules
   chose a client lr of 0.20–0.26 (v1 used 0.05); FedAdam chose a small client lr (0.014) with a
   large server lr (0.28). On the tuning seed, FedAvg, FedNova and FedProx score higher still at
   lr 0.4 and 0.6, so the search range clipped their optimum (Section 9.10).
2. **FedAdam is the best strategy on the random protocol (R = 100).** It ranked first in 20 of 20
   partition × seed blocks (mean rank 1.40), +0.034 macro-F1 above FedAvg on average (bootstrap
   95% CI [0.024, 0.046]; Wilcoxon, Holm p = 1.3 × 10⁻⁵), and it stays ahead of FedAvg and FedProx
   run at the larger lr 0.6 in 17 of 20 blocks (Section 9.11). Its gain is largest under attack-type exclusivity (P3:
   0.842 vs 0.778) and smallest under IID (P0: 0.967 vs 0.956).
3. **FedProx, FedNova and SCAFFOLD are not distinguishable from FedAvg overall.** Tuned FedProx
   chose μ = 0.0012 and behaves as FedAvg. FedNova is +0.010 (n.s.) overall but clearly better
   under P3 (0.822 vs 0.778). **SCAFFOLD is the best rule under P3 (0.871)** but is **unstable at the
   tuned lr**: it diverged in 5 of 20 random-protocol runs and 9 of 20 chronological runs.
4. **Attack-type exclusivity (P3) is the hardest partition** for every strategy; FedAvg loses BFA
   entirely there (recall 0.00). The gap to centralized grows with label skew (realized JSD) for
   6 of 8 strategies, but not with quantity skew (shard-size CV, never significant). In the α sweep
   FedAvg's score moves little (0.923–0.956) until α = 0.1.
5. **Federation clearly beats local-only training under skew.** Local-only clients reach 0.214 mean
   test macro-F1 under P3 (best client 0.320), against 0.778 for FedAvg; under P0 they are as good
   as FedAvg (0.950 vs 0.956).
6. **Without attackers, the robust rules lose substantially under skew** (random P3: Krum 0.143,
   Median 0.535, Trimmed Mean 0.570 vs FedAvg 0.778; all three significantly worse than FedAvg) and
   match FedAvg only under P0.
7. **Poisoning (f = 2 of 8):** on chronological P4 the attacks barely move FedAvg and the robust
   rules do not protect better (Krum's attack FNR rises to 0.58 under targeted relabeling). On
   random P3 sign flipping cuts FedAvg from 0.778 to 0.497 (two runs diverged), most when a
   malicious controller is the **sole holder** of an attack family (0.371 vs 0.688); the robust
   rules are not better under attack, and targeted relabeling raises attack FNR for every rule.
8. **The chronological protocol is dominated by distribution shift.** Centralized reaches 0.519
   (BFA recall 0.005, Probe 0.31), and most strategies sit at 0.51–0.53 in every partition except
   P3. Strategy rankings there are not informative.
9. **Instability is common at the tuned learning rates and is now flagged automatically.** Besides
   SCAFFOLD, one random-protocol centralized run (seed 4) and two chronological centralized runs
   collapsed to predicting one class by their final epoch. Validation-based round selection
   protects most headline numbers, but the random centralized mean drops from 0.972 (4 stable
   seeds) to 0.938 ± 0.075 with seed 4; T95 and the gap regression are reported both ways.
10. **The larger learning rate is not better overall.** FedAvg / FedProx at lr 0.6 on the full
    random grid are not significantly different from their frozen versions (−0.006 / +0.006); they
    gain about 0.08 under P3 but lose under P0 / P1 and diverge in 4 of 40 runs. FedAdam still beats
    both in 17 of 20 blocks (+0.040 / +0.030).

---

## 2. Environment and reproducibility

### Hardware and OS

- One laptop CPU: **Intel Core i7-7600U @ 2.80 GHz, 2 cores / 4 threads**. WSL reports 7 GB RAM;
  the Windows host's total RAM is not recorded. No GPU was used (CPU build of PyTorch).
- **Windows** host with a native venv (`.venv`, Python 3.13.17) for unit tests, and **WSL 2**
  (Ubuntu 26.04.1 LTS, kernel 6.18.54.1-microsoft-standard-WSL2) with venv `~/venvs/fedsdn`
  (Python 3.13.18, same version as Kaggle). **All v2 runs were launched from the WSL venv.**

### Software (exact pins, `requirements.txt`)

| package | version |
|---|---|
| Python | 3.13.18 (WSL / Kaggle), 3.13.17 (Windows) |
| torch | 2.15.0+cpu |
| numpy | 2.5.4 |
| pandas | 3.1.1 |
| pyarrow | 26.0.1 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| scikit-posthocs | 0.17.1 |
| statsmodels | 0.15.0 (installed with scikit-posthocs; used for the OLS regression) |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| pytest | 9.2.0 |
| flwr[simulation] | 1.43.0 (used only for the Flower validation, Section 15) |
| ray | 2.55.1 (WSL only; used only for the Flower validation, Section 15) |

Full freezes: `requirements-lock.txt` (Windows), `requirements-lock-wsl.txt` (WSL).

### Simulation design

- `src/fedsdn/sim.py`: plain PyTorch, **no Flower and no Ray**. In each round the K = 8 clients
  are trained **one after another in the same process**, starting from the same global weights;
  the server rule then combines the K flattened updates.
- `scripts/run_study.py` runs one suite (`tuning`, `random`, `chrono`, `sweep`, `local`,
  `poison_random`, `lr_boundary`, `lr_sensitivity`) with a pool of **4 worker processes**
  (`multiprocessing`, spawn context, `torch.set_num_threads(1)` per worker).
- Resumability: each finished run writes `<run_id>_rounds.csv` and then `<run_id>.json`
  (atomic writes) under `results/<suite>/runs/`. The JSON is the "done" marker, and finished runs
  are skipped on restart. After a suite, `results/<suite>/summary.csv` and `rounds.csv` are
  collected. Progress for all suites is logged to `results/progress.log`.
- Every run summary carries the hyperparameters actually used and two failure flags:
  `diverged` (non-finite weights at any round) and `collapsed` (Section 7). The suite end line in
  `progress.log` reports both counts.
- Paths come from `FEDSDN_DATA` (default `./data`) and `FEDSDN_RESULTS` (default `./results`).

### Commit for each stage

| stage | commit | date (+05:00) |
|---|---|---|
| Project skeleton, pins, original rules (incl. 60 s gap split) | `d3d276c` | 2026-11-05 22:51 |
| WSL environment, Flower smoke test | `acfad95` | 2026-11-06 07:56 |
| InSDN preprocessing, per-family minute-cut split | `2f6d43d` | 2026-11-06 08:25 |
| Partitioning P0–P4, federated scaling | `5fa6680` | 2026-11-06 08:38 |
| v1 study (untuned, 7 strategies, 3 seeds), statistics, diagrams | `1792d65` … `51eb753` | 2026-11-06 09:33–11:22 |
| Flower validation (Section 15) | `6c241e8` | 2026-11-06 11:32 |
| v1 write-up; Flower section | `80fd65f`, `6de8a9d` | 2026-11-06 11:42, 14:11 |
| **v2:** SCAFFOLD, local-only baseline, `run_study.py` | `871aefa` | 2026-11-06 14:21 |
| **v2:** tuning frozen (`configs/tuning/insdn.yaml`); v2 paper outputs | `f3ba4f2` | 2026-11-06 14:53 |
| **v2:** lr boundary suite | `f83f1c1` | 2026-11-06 17:00 |
| **v2:** collapse flag, lr sensitivity suite, statistics, this write-up | see `git log` | 2026-11-06 |

Every v2 suite ran on code identical to `f3ba4f2` for the simulation (`sim.py`) except two
additions made between suites: the `lr_boundary` suite (runner only) and the `collapsed` flag plus
a per-round `top_pred_share` column in `sim.py` (added after the main suites; the flag was
backfilled into the earlier summaries by `scripts/add_collapse_flag.py` from the logged per-class
recalls, so it is computed the same way for every run). Neither changes any trained model.

Data fingerprints: `flows.parquet` SHA-256 `fbf3ad7570b8…a52c` (chronological). Raw CSV SHA-256s
are stored in `data/processed/insdn/meta.json`. Every partition sidecar stores the
`flows.parquet` hash, and the partition scripts refuse to run on a mismatch.

### Commands to reproduce everything

```bash
# WSL venv (see CLAUDE.md for creating it), from the repo root
source ~/venvs/fedsdn/bin/activate
export FEDSDN_DATA=./data FEDSDN_RESULTS=./results PYTHONPATH=src
# raw CSVs in data/raw/insdn/{Normal_data,OVS,metasploitable-2}.csv

python -m pytest                                              # 53 unit tests
python scripts/build_dataset.py --dataset insdn               # processed/insdn
python scripts/make_partitions.py --dataset insdn --K 8 --seeds 0-4      # chrono: 10 settings x 5 seeds
python scripts/make_random_split.py --dataset insdn --seed 0  # processed/insdn_random
python scripts/make_partitions.py --dataset insdn --variant random --seeds 0-4 \
    --settings P0 P1_a0.5 P3_m1 P4_a0.5b0.5 P1_a0.1 P1_a1 P1_a10 P2_b0.5
python scripts/make_partitions.py --dataset insdn --variant random --seeds 99 \
    --settings P1_a0.5 P3_m1 P4_a0.5b0.5                      # tuning partitions
python scripts/run_study.py --suite tuning      # 300 runs; writes configs/tuning/insdn.yaml
python scripts/run_study.py --suite random      # 165 runs
python scripts/run_study.py --suite chrono      # 205 runs
python scripts/run_study.py --suite sweep       # 40 runs
python scripts/run_study.py --suite local       # 20 runs
python scripts/run_study.py --suite poison_random   # 40 runs
python scripts/run_study.py --suite lr_boundary     # 24 runs (validation only)
python scripts/run_study.py --suite lr_sensitivity  # 40 runs
python scripts/add_collapse_flag.py             # only needed for summaries written before the flag existed
python scripts/paper_outputs.py                 # results/paper_tables/*.csv, results/paper_figures/fig1-6, 8-12
python scripts/paper_diagrams.py                # fig0_system_architecture, fig_model_architecture, fig_pipeline
python scripts/flower_validation.py --plot-only # fig7 + flower_validation.csv from results/flower_validation/
```

Any suite accepts `--dry` (2 rounds, one run per kind/strategy/attack → `results/dry_v2/`, never
used for outputs). Runs are deterministic given the code and data: a round never depends on R, so
an R = 30 run equals the first 30 rounds of the R = 100 run with the same seed (used in
Sections 9.1 and 9.5).

---

## 3. Dataset and preprocessing (InSDN)

### Raw data

Three CSV files, 84 CICFlowMeter columns, 343,889 flows in total.

| file | README count | observed | difference |
|---|---|---|---|
| Normal_data.csv | 68,424 | 68,424 | 0 |
| OVS.csv | 136,743 | 138,722 | +1,979 |
| metasploitable-2.csv | 138,772 | 136,743 | −2,029 |

The README counts for OVS and metasploitable-2 appear to be swapped; even after swapping them, the
total differs by 50 rows (README 343,939 vs observed 343,889). Expected, observed and difference
are recorded in `meta.json`.

### Steps, in order, with the decision and its reason

1. **Load.** All three CSVs, each row given a stable `row_id` in (file name, file row) order, used
   as the join key for partitions.
2. **Clean strings and parse time.** Column names and labels have stray whitespace (`"DDoS "`),
   so both were stripped. `Timestamp` is day-first and comes in exactly two formats: 24-hour
   `D/M/YYYY H:MM` at minute resolution (258,078 rows) and 12-hour `DD/MM/YYYY hh:MM:SS AM|PM`
   at second resolution (85,811 rows). Both were parsed explicitly, with the resolution (60 s or
   1 s) recorded per row. The source has no time zone, so times were read as UTC. 0 rows were
   unparsed. Time range: 2019-12-25 17:16:48 to 2020-02-07 19:53:00.
3. **Invalid rows.** Rows with NaN or infinite feature values, or missing label / IP / timestamp,
   were dropped. There were 0 such rows (checked in code anyway).
4. **Identifiers removed from the features**: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`,
   `Timestamp`. **`Dst Port` was kept as a numeric feature.** Reason: IPs, flow IDs and timestamps
   allow shortcut learning and do not generalize across domains (paper III-C). The source port is
   an ephemeral, per-connection value with the same problem. The destination port identifies the
   targeted service, which is real information a controller sees. The timestamp and IP addresses
   were kept as **metadata only** (for the split), never as model inputs.
5. **Deduplication on feature columns + label** (after step 4, so rows that differ only in
   identifiers count as duplicates):
   - First, any group of rows with identical features but **different labels** is dropped
     entirely, because no classifier can be right on both and keeping one label would be
     arbitrary. InSDN had **0** such rows.
   - Then, among exact feature + label duplicates, the **earliest** flow (by `ts`, then `row_id`)
     is kept, so that the split in step 9 does not place copies of the same flow on both sides.
   - **125,204** duplicate rows were dropped: DDoS 120,422, DoS 3,176, Normal 1,459, Probe 114,
     BFA 24, Web-Attack 9. **218,685** rows remain.
   - Class counts after dedup: Normal 66,965; Probe 98,015; DoS 50,440; DDoS 1,520; BFA 1,381;
     Web-Attack 183; BOTNET 164; U2R 17. (DDoS shrinks from about 122k to 1,520.)
6. **Labels.** `y_bin` (Normal = 0, every attack = 1) and `family` (attack family) for the
   multi-class task.
7. **Rare-class rule** (paper III-C): families with < 500 flows after dedup are merged into
   `Rare`. If `Rare` still has < 500 flows, its rows get `y_multi = −1`: they are excluded from
   the multi-class task but stay as attacks in the binary task. InSDN: Rare = BOTNET 164 +
   Web-Attack 183 + U2R 17 = **364 < 500 → excluded**. DDoS (1,520) and BFA (1,381) stay as their
   own classes. Final multi-class index: Normal 0, BFA 1, DDoS 2, DoS 3, Probe 4.
8. **Class cap 20,000** (paper III-C, to keep the study feasible): classes above the cap are thinned
   uniformly in time order (sort by `ts`, `row_id`; keep positions ⌊i·n/20000⌋). DoS 50,440 →
   20,000; Normal 66,965 → 20,000; Probe 98,015 → 20,000. 155,420 rows were removed by the cap.
   They are kept in `flows.parquet` with `split = capped`, so all counts can be checked.
9. **Train / test split per family, between minutes** (non-capped rows only).
   - **Planned rule, and why it was replaced.** The paper (and the first version of `CLAUDE.md`,
     commit `d3d276c`) specified: per class, the earliest 80% go to train and the latest 20% to
     test, discarding flows within ±60 s of the boundary. Most InSDN rows have only
     minute-resolution timestamps, and each attack family occupies very few distinct minutes
     (BFA 14, DDoS 9, BOTNET 3, DoS 20). A ±60 s window around the boundary therefore covers whole
     minutes holding a large share of the family. The repository does not record the losses that
     were seen when this rule was rejected; recomputing the rule on the same post-cap rows gives
     **[computed for this write-up, per family]**:

     | family | flows | discarded in gap | train | test |
     |---|---|---|---|---|
     | BFA | 1,381 | 822 (59.5%) | 559 | **0** |
     | BOTNET | 164 | 114 (69.5%) | 50 | **0** |
     | DDoS | 1,520 | 999 (65.7%) | 498 | 23 |
     | DoS | 20,000 | 7,979 (39.9%) | 9,857 | 2,164 |
     | Probe | 20,000 | 5,643 (28.2%) | 11,652 | 2,705 |
     | Normal | 20,000 | 224 (1.1%) | 15,900 | 3,876 |
     | Web-Attack | 183 | 28 (15.3%) | 121 | 34 |
     | U2R | 17 | 3 (17.6%) | 12 | 2 |

     The gap rule would have discarded 28–70% of most attack families and left BFA and BOTNET
     with no test flows at all. It was replaced in commit `2f6d43d`.
   - **Rule used.** Floor every `ts` to its minute. A *minute cut* splits **between** two
     consecutive minutes (train = minutes ≤ cut), so no minute appears on both sides and no flow
     is discarded. For each family, the first rule that succeeds is used (ties → earliest cut):
     - **(a) `minute_cut`**: among cuts giving a test share in [10%, 35%] and ≥ 100 test flows,
       take the one closest to 20%.
     - **(b) `minute_cut_relaxed`**: among cuts with ≥ 100 test flows and test share ≤ 50%, take
       the one closest to the range [10%, 35%].
     - **(c) `grouped_src_dst_pair`**: whole unordered source–destination address pairs go to
       test, under the constraints of (a), using an exact subset-sum over pair sizes (ties → the
       smaller test set, pairs taken deterministically in name order). Pairs per flow is
       recorded, and > 0.5 is flagged as degenerate grouping.
     - **(d) `unsplit_all_train`**: the whole family goes to train.
     - Rare families (BOTNET, Web-Attack, U2R) are split separately, without the 100-flow minimum,
       and in (b) they also need a test share ≥ 10%.
   - **Result per family** (accepted):

     | family | method | n | train | test | test share | minutes train / test / shared | notes |
     |---|---|---|---|---|---|---|---|
     | Normal | (a) minute cut | 20,000 | 15,987 | 4,013 | 20.1% | 231 / 362 / 0 | |
     | Probe | (a) minute cut | 20,000 | 16,166 | 3,834 | 19.2% | 23 / 30 / 0 | |
     | DoS | (a) minute cut | 20,000 | 17,836 | 2,164 | 10.8% | 3 / 17 / 0 | the only acceptable cut |
     | DDoS | (b) relaxed cut | 1,520 | 871 | 649 | 42.7% | 7 / 2 / 0 | no cut in [10%, 35%] |
     | BFA | (c) grouped pair | 1,381 | 1,087 | 294 | 21.3% | 12 / 3 / 1 | 3 pairs; test pair 192.168.3.130 ↔ 200.175.2.130; 0.0022 pairs/flow |
     | Web-Attack | (a) minute cut | 183 | 145 | 38 | 20.8% | 4 / 4 / 0 | rare (y = −1) |
     | U2R | (a) minute cut | 17 | 15 | 2 | 11.8% | 6 / 1 / 0 | rare (y = −1) |
     | BOTNET | (d) all train | 164 | 164 | 0 | 0% | 3 / 0 / 0 | rare (y = −1) |

   - **DDoS:** its source addresses are spoofed (one address pair per flow), so a grouped split
     (c) would be a random flow-level split in disguise. The relaxed minute cut (b) at 42.7% was
     preferred and accepted, so (c) is never reached for DDoS.
   - **BFA:** no minute cut met (a) or (b), so the grouped split was used. BFA has only 3 address
     pairs (0.0022 pairs per flow, not degenerate), and the test set is one whole pair. This is a
     strict test of generalization to an unseen attacker/target pair, and it is the main reason
     for BFA's zero test recall on the chronological protocol.
   - **BOTNET:** no rule produced an acceptable test set (3 minutes in total), so the whole family
     went to train. It is excluded from the multi-class task anyway (step 7).
10. **Features.** 12 columns that are constant over the training pool were dropped: `Fwd PSH Flags`,
    `Fwd URG Flags`, `CWE Flag Count`, `ECE Flag Cnt`, `Fwd Byts/b Avg`, `Fwd Pkts/b Avg`,
    `Fwd Blk Rate Avg`, `Bwd Byts/b Avg`, `Bwd Pkts/b Avg`, `Bwd Blk Rate Avg` (all 0),
    `Init Fwd Win Byts` (−1), `Fwd Seg Size Min` (0). `Protocol` was one-hot encoded with the
    categories seen in the training pool (0, 6, 17). **Final feature count: 68** (65 numeric,
    including `Dst Port`, + 3 protocol indicators). After this step, 3 pairs of non-capped rows
    became identical in feature space; this is recorded in `meta.json` and was not acted on.
11. **No scaling in preprocessing.** Standardization is done inside each federation (Section 5).
12. **Test set never seen by clients**; each client holds out 10% of its shard for validation
    (Section 4); class imbalance is handled with class-weighted loss, never oversampling.

### Final counts

Chronological protocol (`data/processed/insdn`):

| class | train | test |
|---|---|---|
| Normal | 15,987 | 4,013 |
| BFA | 1,087 | 294 |
| DDoS | 871 | 649 |
| DoS | 17,836 | 2,164 |
| Probe | 16,166 | 3,834 |
| Rare (excluded, y = −1) | 324 | 40 |
| **multi-class total** | **51,947** | **10,954** |
| all rows (incl. Rare) | 52,271 | 10,994 |

Binary counts (not used in the reduced study): train 15,987 benign / 36,284 attack; test 4,013 /
6,981.

Random protocol (`data/processed/insdn_random`, `scripts/make_random_split.py`): the same 63,265
non-capped rows, re-split 80/20 at random, stratified by partition family (Rare as one family),
seed 0:

| class | train | test |
|---|---|---|
| Normal | 16,000 | 4,000 |
| BFA | 1,105 | 276 |
| DDoS | 1,216 | 304 |
| DoS | 16,000 | 4,000 |
| Probe | 16,000 | 4,000 |
| Rare (excluded) | 291 | 73 |
| **multi-class total** | **50,321** | **12,580** |

---

## 4. Partitioning

Implemented in `src/fedsdn/data/partition.py`. Partitions cover training-pool rows only and are
built on the *partition family*, i.e. `family` with BOTNET / Web-Attack / U2R mapped to one family
`Rare`. This gives 6 families (Normal + 5 attack families, Rare included), so the same shards
would serve the binary task. In the multi-class task, Rare rows stay in the shards but are masked
out of training and evaluation.

"Dealing" = shuffle the rows, then hand out consecutive chunks whose sizes are the proportions
rounded by the largest-remainder method (sizes sum exactly to the row count). Each (setting, seed)
has its own RNG stream (`default_rng([seed, crc32(tag)])`).

### Schemes

- **P0 (IID):** shuffle all rows, deal into K equal shards.
- **P1 (Dirichlet label skew):** for each family c, draw q_c ~ Dir_K(α) and deal that family's
  rows by q_c. α ∈ {0.1, 0.5, 1, 10} built and used (α = 0.5 in the main grid; 0.1, 1, 10 in the
  sweep).
- **P2 (quantity skew):** one s ~ Dir_K(β); all rows dealt by s, so the class mix of each shard
  matches the global mix in expectation. β ∈ {0.5, 1} built; **β = 0.5 used in the sweep**.
- **P3 (attack-type exclusivity):** benign rows dealt equally over all K clients; the attack
  families are put in a seeded random order and assigned round-robin, with
  m_eff = max(m, ⌈F/K⌉) families per client (F = 5 attack families, K = 8 → m_eff = 1 for m = 1).
  Families are assigned with replacement, so every family has at least one holder; a family
  is dealt equally among its holders. m ∈ {1, 2} built; **m = 1 used**.
- **P4 (combined skew, product reweighting):** an approved addition not specified in the paper
  beyond "α = β = 0.5". One s ~ Dir_K(0.5) for shard sizes, and per family q_c ~ Dir_K(0.5).
  Family c is dealt by q̃_{c,k} = q_{c,k}·s_k / Σ_j q_{c,j}·s_j. With uniform s this reduces to
  P1; with uniform q it reduces to P2 (in expectation), as checked by a unit test. **Used.**

Built: chronological protocol, all 10 settings × seeds 0–4 (50 partitions). Random protocol:
P0, P1 α ∈ {0.1, 0.5, 1, 10}, P2 β = 0.5, P3 m = 1, P4 × seeds 0–4 (40), plus the tuning
partitions P1 α = 0.5, P3 m = 1, P4 with **seed 99** (3), never used outside tuning.

### Minimum shard size

Every shard (validation rows included) must have n_min = 100 rows. Up to 100 fresh draws are
tried. If all fail, the draw with the largest minimum shard is kept and random rows are moved from
the current largest shard. Every used partition met the minimum within 1–3 draws, and **no rows
were moved** in any used partition.

### Validation hold-out

Per client, round(0.1·n_k) rows are held out for validation. They are spread over families in
proportion to the client's family counts (largest remainder) and drawn at random within each
family. Validation rows are excluded from training, from the standardization statistics and from
the class-weight counts.

### Realized heterogeneity (K = 8; ranges over seeds 0–4; `table_partition_heterogeneity.csv` has every seed)

JSD = mean pairwise Jensen–Shannon divergence (base 2) of the clients' family distributions,
computed over whole shards. CV = coefficient of variation of shard sizes (population std).

| protocol | partition | JSD | CV | min shard | max shard | draws |
|---|---|---|---|---|---|---|
| chrono | P0 | 0.000 | 0.000 | 6,533 | 6,534 | 1 |
| chrono | P1 α0.5 | 0.319–0.376 | 0.359–0.743 | 657–2,643 | 9,764–16,964 | 1 |
| chrono | P3 m1 | 0.485–0.531 | 0.609–0.868 | 2,160–2,322 | 10,917–18,164 | 1 |
| chrono | P4 | 0.212–0.419 | 0.915–1.392 | 139–304 | 14,301–28,230 | 1–2 |
| random | P0 | 0.000 | 0.000 | 6,326 | 6,327 | 1 |
| random | P1 α0.1 | 0.660–0.739 | 0.605–1.161 | 158–697 | 10,851–18,789 | 1–3 |
| random | P1 α0.5 | 0.329–0.426 | 0.263–0.720 | 460–3,582 | 9,063–15,273 | 1 |
| random | P1 α1 | 0.146–0.275 | 0.283–0.627 | 1,752–4,315 | 8,924–12,700 | 1 |
| random | P1 α10 | 0.012–0.024 | 0.125–0.229 | 4,651–5,315 | 7,164–9,902 | 1 |
| random | P2 β0.5 | 0.001–0.003 | 0.941–1.389 | 231–464 | 16,249–25,698 | 1–3 |
| random | P3 m1 | 0.482–0.527 | 0.582–0.859 | 2,145–2,291 | 10,000–18,000 | 1 |
| random | P4 | 0.222–0.402 | 0.819–1.288 | 113–258 | 16,364–24,961 | 1–3 |
| random (tuning, seed 99) | P1 α0.5 / P3 m1 / P4 | 0.284 / 0.484 / 0.369 | 0.465 / 0.859 / 1.433 | 2,130 / 2,145 / 213 | 9,590 / 18,000 / 28,423 | 1 / 1 / 2 |

P3 and P1 α = 0.1 have the highest label skew; P2 and P4 the highest quantity skew. Across the
partitions used in the regression (Section 10), JSD and CV are correlated (r = 0.45 with the sweep,
0.73 on the main grid alone).

### P3 per-client family layout and malicious controllers (attack families only; Normal is on all 8)

With 5 attack families (Rare included) over 8 clients and m_eff = 1, three families have two holders
and two have one. A client whose only attack family is Rare trains on **Normal only** in the
multi-class task (Rare is masked). Malicious controllers (poisoning runs) are drawn per seed with
`default_rng([seed, 2024])`, independently of the partition.

| protocol | seed | family → clients | malicious | malicious sole holder of |
|---|---|---|---|---|
| chrono | 0 | BFA {2, 7}; DDoS {3}; DoS {1, 6}; Probe {4}; Rare {0, 5} | {0, 3} | (not run on chrono P3) |
| chrono | 1 | BFA {3}; DDoS {4}; DoS {1, 6}; Probe {2, 7}; Rare {0, 5} | {1, 4} | |
| chrono | 2 | BFA {2, 7}; DDoS {4}; DoS {0, 5}; Probe {3}; Rare {1, 6} | {0, 3} | |
| chrono | 3 | BFA {0, 5}; DDoS {4}; DoS {1, 6}; Probe {2, 7}; Rare {3} | {0, 4} | |
| chrono | 4 | BFA {1, 6}; DDoS {3}; DoS {2, 7}; Probe {4}; Rare {0, 5} | {3, 6} | |
| random | 0 | BFA {2, 7}; DDoS {0, 5}; DoS {3}; Probe {1, 6}; Rare {4} | {0, 3} | **DoS** |
| random | 1 | BFA {3}; DDoS {2, 7}; DoS {0, 5}; Probe {1, 6}; Rare {4} | {1, 4} | none (only Rare, which is masked) |
| random | 2 | BFA {2, 7}; DDoS {4}; DoS {0, 5}; Probe {3}; Rare {1, 6} | {0, 3} | **Probe** |
| random | 3 | BFA {0, 5}; DDoS {1, 6}; DoS {3}; Probe {2, 7}; Rare {4} | {0, 4} | none (only Rare) |
| random | 4 | BFA {3}; DDoS {2, 7}; DoS {4}; Probe {0, 5}; Rare {1, 6} | {3, 6} | **BFA** |

Normal is spread equally (about 2,000 rows per client).

---

## 5. Model and training

- **Model** (`src/fedsdn/models.py`): MLP 68 → 128 → 64 → 5, ReLU, dropout 0.2 after each hidden
  layer, logits output with `CrossEntropyLoss`. **17,413 parameters** (8,832 + 8,256 + 325),
  68 KB in float32. The secondary 1D-CNN was not run.
- **Local optimizer:** plain mini-batch SGD, **momentum 0, no weight decay**, batch size 256,
  E = 1 local epoch per round, **client learning rate tuned per strategy** (Section 5.1). Local
  steps per round τ_k = E·⌈n_k / 256⌉, where n_k is the client's training rows after masking
  Rare. A fresh optimizer is created every round, so there is no optimizer state across rounds.
- **Class weights (global balanced):** each client reports its training-row class counts
  (validation excluded, Rare masked); the server sums them and sets w_c = n_train / (C·n_c), with
  C = 5. Computed once per run, before round 1; the same weights go to every client.
- **Federated standardization** (`src/fedsdn/data/scaling.py`): each client sends count, per-feature
  sum and sum of squares over its training rows (validation excluded); the server computes the
  global mean and population std in float64. A feature with variance ≤ 1e-12·E[x²] is treated as
  constant and gets std 1. The same mean / std are applied to every row (train, validation, test).
  A unit test checks that this equals pooled NumPy standardization.
- **Rare rows (y = −1)** are dropped from each client's training rows and from validation and test
  evaluation (equivalent to masking). They are included in the standardization statistics.
- **Rounds:** R = 100 on the random protocol (main grid, sweep, random poisoning, sensitivity),
  R = 30 on the chronological protocol (grid and poisoning) and for tuning / the boundary check.
- **Centralized baseline:** the same model, optimizer, batch size, class weighting and
  standardization (one "client" with the whole pool), with its own tuned lr. It holds out a
  stratified 10% of the training pool as validation, trains for as many epochs as the protocol has
  rounds (100 random, 30 chronological) and reports the test score at the best-validation epoch.
- **Local-only baseline** (`kind = local` in `sim.py`): every client trains alone on its own
  training rows for 100 epochs (no communication), with standardization and class weights from its
  own rows only, the centralized tuned lr, and the same initial model for all clients. Each client
  picks its epoch by its own validation macro-F1 (classes present in its validation set) and is
  scored on the global test set; the run reports the mean, worst and best client.
- **Seeds 0–4** (tuning: 99). The seed selects the partition file and sets `torch.manual_seed`
  (model initialization and dropout). Batch order uses a separate generator per (seed, round,
  client): `seed·1,000,003 + round·101 + client`.

### 5.1 Hyperparameter tuning (paper Section III-H)

- **Search:** 12 random-search trials per strategy, log-uniform. The 12 client learning rates in
  [1e-3, 3e-1] were **drawn once** (`default_rng(99)`) and shared by every strategy and the
  centralized baseline (trial i uses lr_i everywhere): 0.0179, 0.0251, 0.0185, **0.2560**, 0.0334,
  0.0256, 0.0051, 0.0236, 0.0144, 0.0324, **0.2018**, 0.0041. FedProx also draws μ in [1e-3, 1]
  (`rng [99, 1]`); FedAdam also draws the server lr in [1e-2, 1] (`rng [99, 2]`) and ε in
  [1e-3, 1e-1] (`rng [99, 3]`), one value per trial. Not tuned: β1 = 0.9, β2 = 0.99 (FedAdam),
  f̂ = 2, SCAFFOLD global step 1, E, B.
- **Setting:** random protocol, multi-class, R = 30, tuning seed 99 with its own partitions
  (P1 α = 0.5, P3 m = 1, P4); the centralized baseline is tuned on its own 10% validation split.
- **Score:** mean over the 3 partitions of the best pooled validation macro-F1 over rounds (the
  same validation-based selection as the headline metric). **The test set is never read.** A trial
  in which any run diverged (non-finite weights) is ineligible; ties → earliest trial.
- **Frozen** in `configs/tuning/insdn.yaml` (written by `run_study.py --suite tuning`, commit
  `f3ba4f2`) and used unchanged by every later run.

| strategy | client lr | other | best score | median | min–max over 12 trials | 2nd best | diverged trials |
|---|---|---|---|---|---|---|---|
| Centralized | 0.2018 | — | 0.919 | 0.847 | 0.738–0.919 | 0.855 | 0 |
| SCAFFOLD | 0.2018 | — | 0.880 | 0.780 | 0.626–0.880 | 0.793 | 1 (lr 0.256) |
| FedAdam | 0.0144 | η_s 0.2848, ε 0.00667 | 0.878 | 0.793 | 0.436–0.878 | 0.862 | 0 |
| FedAvg | 0.2560 | — | 0.848 | 0.762 | 0.593–0.848 | 0.842 | 0 |
| FedNova | 0.2560 | — | 0.843 | 0.747 | 0.626–0.843 | 0.842 | 0 |
| FedProx | 0.2018 | μ 0.00117 | 0.842 | 0.760 | 0.593–0.842 | 0.804 | 0 |
| Trimmed Mean | 0.2560 | — | 0.613 | 0.540 | 0.508–0.613 | 0.603 | 0 |
| Median | 0.2560 | — | 0.594 | 0.543 | 0.478–0.594 | 0.580 | 0 |
| Krum | 0.2018 | — | 0.402 | 0.366 | 0.179–0.402 | 0.396 | 0 |

- Every plain-SGD rule picked one of the two large learning rates (0.20 or 0.26); only 2 of the 12
  shared draws were above 0.04, so the search was coarse exactly where the optimum lies. The
  boundary check (Section 9.10) shows that FedAvg, FedNova and FedProx score higher still at
  lr 0.4–0.6.
- FedAdam is the exception: it prefers a small client lr with a large server lr, and it has the
  widest spread over trials (0.44–0.88), i.e. it is the most tuning-sensitive.
- SCAFFOLD's lr 0.256 trial scored 0.8796 but was ineligible (one run diverged); lr 0.2018 scored
  0.8795. The frozen lr is therefore right at SCAFFOLD's stability edge, which shows up as
  divergence in the main grid (Section 9.9).
- FedProx's tuned μ = 0.0012 is so small that FedProx is effectively FedAvg; its best trial has
  lr 0.2018, not FedAvg's 0.256, because the μ drawn with lr 0.256 (trial 3) was larger.
- Robust rules were tuned without attackers (their tuning runs are honest), so the poisoning runs
  use clean-tuned values.
- Spread per trial: `results/tuning/trials.csv`, `tuning_summary.csv`;
  `paper_tables/table_tuning_trials.csv`, `table_tuning_chosen.csv`; figure `fig8_tuning.png`.

---

## 6. Aggregation strategies

Implemented in `src/fedsdn/strategies/aggregate.py` (server rules) and `src/fedsdn/sim.py` (client
side). Each client returns its update Δ_k = w_k − w (flattened). The server computes an aggregate
Δ and sets w ← w + Δ. p_k = n_k / Σ_j n_j, with n_k = the client's training rows (validation
excluded, Rare masked).

| strategy | rule (in words) | hyperparameters (tuned values) | weighting |
|---|---|---|---|
| **FedAvg** | Δ = Σ p_k Δ_k | client lr 0.256 | by sample count |
| **FedProx** | Same server rule as FedAvg. Each client adds (μ/2)‖w − w_global‖² to its local loss. | client lr 0.202, μ = 0.00117 | by sample count |
| **FedNova** | d_k = Δ_k / τ_k; Δ = τ_eff·Σ p_k d_k, τ_eff = Σ p_k τ_k. Equals FedAvg when all τ_k are equal. | client lr 0.256 (plain SGD so τ_k is exact) | by sample count |
| **FedAdam** | Pseudo-gradient g = FedAvg Δ; m ← β1·m + (1−β1)·g; v ← β2·v + (1−β2)·g²; Δ = η_s·m / (√v + ε). m, v start at 0; no bias correction (as in the paper's formula). | client lr 0.0144, η_s = 0.2848, ε = 0.00667, β1 = 0.9, β2 = 0.99 | by sample count |
| **SCAFFOLD** | Option II of Karimireddy et al., global step 1, full participation. Client k trains with gradient + (c − c_k); after τ_k steps at rate η: c_k⁺ = c_k − c − Δ_k/(τ_k·η). Server: Δ = Σ p_k Δ_k and c ← c + Σ p_k (c_k⁺ − c_k), so c = Σ p_k c_k at all times. | client lr 0.202 | by sample count (updates and variates) |
| **Krum** | For each update, sum the squared distances to its K − f̂ − 2 = 4 nearest other updates; the update with the smallest sum becomes the global update. | f̂ = 2; client lr 0.202 | equal (one update selected) |
| **Median** | Coordinate-wise median of the 8 updates (even K: mean of the two middle values). | client lr 0.256 | equal |
| **Trimmed Mean** | Per coordinate, drop the f̂ = 2 largest and 2 smallest values and average the remaining 4. | f̂ = 2; client lr 0.256 | equal |

- **SCAFFOLD details.** Client variates c_k persist across rounds in the server-side `Scaffold`
  object, keyed by client id (the plain loop has no ephemeral client objects). Messages carry
  (Δ_k, Δc_k) up and (w, c) down, so **SCAFFOLD's bytes per round are twice FedAvg's** (2.23 MB vs
  1.11 MB). Variates are aggregated with the same n_k weights as the updates, so that SCAFFOLD with
  frozen zero variates is exactly FedAvg (the published algorithm averages uniformly over sampled
  clients; with full participation and n_k weights, c stays the weighted mean of the c_k).
- f̂ = ⌊K/4⌋ = 2 in **every** condition, honest ones included, as the paper specifies. K = 8
  satisfies Krum (8 ≥ 2f̂ + 3 = 7) and Trimmed Mean (8 > 2f̂ = 4).
- **Poisoning** (in `sim.py`). Malicious clients are fixed per seed (Section 4 table).
  - *sign flip*: the client sends −Δ_k.
  - *targeted*: before training, every attack-labelled training row on the malicious client is
    relabelled Normal.
  - Malicious clients' validation sets keep their true labels.
  - Random label flipping was not run.

### Tests (`tests/`, 53 tests, all passing; re-run for this write-up)

- `test_aggregate.py`:
  - FedNova equals FedAvg when all τ_k are equal; FedNova differs with unequal steps (hand-computed
    4/3);
  - Median and Trimmed Mean on known vectors; Krum rejects an outlier;
  - **SCAFFOLD equals FedAvg with control variates frozen at zero** (3 clients of unequal size,
    3 rounds, real MLP training through `sim.local_train`; weights bitwise equal, variates stay 0);
  - SCAFFOLD with live variates departs from FedAvg and keeps c = Σ p_k c_k;
  - the SCAFFOLD client-variate formula (hand-computed);
  - the collapse flag (Section 7).
- `test_partition.py`: largest-remainder sizes; Rare mapping; every row assigned exactly once and
  deterministically; P4 reduces to P1 / P2; P4 combines both skews; P0 balance; smaller α → higher
  JSD; P2 keeps the label mix; P3 holders; minimum-shard retry and row moves; 10% stratified
  validation; partition file writing; **federated scaling equals pooled NumPy** and excludes
  validation rows.
- `test_preprocess.py`: conflict groups and earliest duplicate; the rare rule; the time-ordered
  cap; minute-cut rules; grouped fallback; relaxed cut; pairs-per-flow flag; rare families and the
  all-train fallback; both timestamp formats; an end-to-end check.

---

## 7. Evaluation

### Protocols

- **Chronological (primary in the paper):** the leakage-aware per-family minute-cut split of
  Section 3 (`processed/insdn`, `partitions/insdn`). R = 30.
- **Random (secondary):** a random stratified 80/20 split of the same non-capped rows, stratified by
  partition family, seed 0 (`processed/insdn_random`, `partitions/insdn_random`). R = 100.

**Why the random protocol exists.** The centralized baseline on the chronological split reaches
only about 0.52 macro-F1. The v1 diagnostics (`scripts/diag_central.py`, seed 0) showed the model
was not underfitting (training macro-F1 0.88) and not undertrained (100 epochs still gave 0.52),
while a random split of the same rows gave 0.95; Section 9.1 repeats the comparison with v2
settings. The gap is distribution shift between train and test periods / address pairs (BFA's test
set is one held-out address pair). The random protocol makes strategy differences visible, but it
has the flow-level leakage the paper warns about: **its absolute scores are inflated and must not
be compared with chronological scores.** It is used to rank strategies against each other, and all
statistical tests use it.

### Metrics (computed every round on the global test set; multi-class, Rare masked)

- **Macro-F1** over the 5 classes (primary). Also accuracy, **per-class recall**, FPR (Normal flows
  predicted as any attack) and **attack FNR** (attack flows predicted as Normal; used for
  poisoning). Rare-class recall = recall of the single class with the fewest training rows
  (DDoS chronological, BFA random).
- **Validation macro-F1** (global): on the pooled validation sets of all clients.
  **Worst-client validation macro-F1:** per client, macro-F1 over the classes present in its own
  validation set; minimum over clients.
- **Validation-selected round (headline):** the round with the best global validation macro-F1
  (ties → earliest). The run reports test metrics at that round; the test set is never used for
  selection. Also reported: the **final round** and **AUC** = mean test macro-F1 over rounds 1..R.
- **T95:** the first round whose test macro-F1 is ≥ 0.95 × the protocol's mean centralized
  validation-selected test macro-F1 (centralized trained for as many epochs as the protocol has
  rounds). Because some centralized runs collapsed (Section 9.9), T95 is reported **twice**:
  - with all 5 centralized seeds: random 0.95 × 0.938 = **0.891**; chronological 0.95 × 0.519 =
    **0.493**;
  - with the stable seeds only (neither diverged nor collapsed): random seeds 0–3,
    0.95 × 0.972 = **0.923**; chronological seeds 0, 1, 3, 0.95 × 0.522 = **0.496**.

  Runs that never reach the threshold are censored. Tables report the mean T95 over the seeds that
  reach it, plus the number censored.
- **Bytes per round** (derived): params × 4 B × 2K = 17,413 × 4 × 16 = **1,114,432 B (1.11 MB)**
  for every strategy except **SCAFFOLD, 2,228,864 B (2.23 MB)**. **MB to T95** = T95 × bytes per
  round. The Flower validation measured real serialized messages at 1.028 × the derived value
  (Section 15).
- **Failure flags** (per run, in every summary):
  - `diverged`: the global weights became non-finite at some round;
  - `collapsed`: at the final round, one class has test recall ≥ 0.99 and every other class
    ≤ 0.01, i.e. the model predicts (nearly) one class for every test flow. Diverged runs are also
    collapsed (NaN logits are predicted as class 0). The flag uses only the logged per-class
    recalls, so it was backfilled for runs that finished before it existed; runs made afterwards
    also log the exact share of the most-predicted class per round (`top_pred_share`).
- **Time:** each round's wall-clock time is logged (`wall_s`). Client CPU time and server
  aggregation time were **not** measured separately; with 4 worker processes on 2 physical cores,
  wall time is not a clean cost measure.

---

## 8. Run inventory

| suite | what | runs | elapsed (4 workers) | where |
|---|---|---|---|---|
| Dry runs | `--dry`, R = 2, one run per kind / strategy / attack per suite | 46 | about 5 min (14:16–14:21) | `results/dry_v2/` (never used for outputs) |
| `tuning` | 9 (8 strategies + centralized) × 12 trials × 3 partitions (centralized: × 1), seed 99, R = 30 | 300 | 31.2 min (14:22–14:53) | `results/tuning/` |
| `random` | 5 centralized (100 epochs) + 8 strategies × 4 partitions × 5 seeds, R = 100 | 165 | 55.7 min (14:53–15:49) | `results/random/` |
| `chrono` | 5 centralized (30 epochs) + 160 federated + 40 poisoning (P4, 4 rules × 2 attacks × 5 seeds), R = 30 | 205 | 29.5 min (15:49–16:19) | `results/chrono/` |
| `sweep` | FedAvg, FedAdam × P1 α 0.1 / 1 / 10, P2 β 0.5 × 5 seeds, R = 100 | 40 | 13.9 min (16:19–16:33) | `results/sweep/` |
| `local` | local-only, 4 partitions × 5 seeds, 100 epochs per client | 20 | 8.7 min (16:33–16:42) | `results/local/` |
| `poison_random` | random P3 m = 1, 4 rules × 2 attacks × 5 seeds, R = 100 | 40 | 13.0 min (16:42–16:55) | `results/poison_random/` |
| `lr_boundary` | seed 99, FedAvg / FedNova / FedProx / SCAFFOLD × lr 0.4, 0.6 × 3 tuning partitions, R = 30 (validation only) | 24 | 3.3 min (16:55–16:59) | `results/lr_boundary/` |
| `lr_sensitivity` | FedAvg, FedProx at lr 0.6 × 4 partitions × 5 seeds, R = 100 | 40 | 16.0 min (17:10–17:26) | `results/lr_sensitivity/` |
| v1 (archived) | untuned 7-strategy study, Flower validation | — | — | `results/archive_v1/` |

- Mean wall time per run: random R = 100 federated 73–90 s (FedProx slowest), centralized
  100 epochs 88 s; chronological R = 30 federated 35.5 s, poisoning 23 s, centralized 59 s; sweep
  81 s; local-only 101 s; random poisoning 76 s; tuning 24.5 s.
- `results/flower_validation/` is a copy of the archived v1 Flower validation, kept so that
  `flower_validation.py --plot-only` can rebuild fig7.

---

## 9. Results

All values are mean ± std over seeds 0–4, test macro-F1 at the validation-selected round unless
stated. "chrono" = chronological, R = 30. "random" = random split, R = 100. **Do not compare chrono
and random numbers with each other.** Centralized is given both over all 5 seeds and over the
stable seeds (Section 9.9).

### 9.1 Leakage diagnostic (centralized, 30 epochs on both protocols)

The random-protocol values are the first 30 epochs of the 100-epoch runs, re-selected by validation
(identical to 30-epoch runs; Section 2).

| protocol | macro-F1 | Normal | BFA | DDoS | DoS | Probe |
|---|---|---|---|---|---|---|
| chronological | 0.519 ± 0.009 | 0.984 | 0.005 | 1.000 | 0.770 | 0.306 |
| random | 0.926 ± 0.070 | 0.988 | 0.805 | 0.997 | 0.965 | 0.944 |

The same model, rows and budget lose about 0.41 macro-F1 when the test set comes from later minutes
/ an unseen address pair; the loss sits in BFA (recall ≈ 0) and Probe (0.31). The random std is
inflated by seed 4, which collapsed after epoch 3 (Section 9.9); without it the random value is
0.956 ± 0.014 **[computed for this write-up from `results/random/rounds.csv`, seeds 0–3, epochs
1–30]**, a gap of 0.44 to the chronological split.

### 9.2 Headline: validation-selected test macro-F1

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.512 ± 0.004 | 0.513 ± 0.004 | **0.675 ± 0.037** | **0.519 ± 0.006** | 0.956 ± 0.004 | 0.938 ± 0.017 | 0.778 ± 0.005 | 0.941 ± 0.046 |
| FedProx | 0.512 ± 0.003 | 0.512 ± 0.004 | 0.673 ± 0.035 | 0.518 ± 0.005 | 0.957 ± 0.004 | 0.929 ± 0.016 | 0.771 ± 0.005 | 0.946 ± 0.045 |
| FedNova | 0.514 ± 0.004 | 0.514 ± 0.005 | 0.545 ± 0.007 | 0.515 ± 0.008 | 0.951 ± 0.011 | 0.943 ± 0.013 | 0.822 ± 0.023 | 0.938 ± 0.017 |
| FedAdam | 0.513 ± 0.002 | **0.516 ± 0.012** | 0.529 ± 0.047 | 0.518 ± 0.006 | **0.967 ± 0.003** | **0.971 ± 0.002** | 0.842 ± 0.026 | **0.969 ± 0.004** |
| SCAFFOLD | **0.526 ± 0.080** | 0.512 ± 0.006 | 0.507 ± 0.059 | 0.485 ± 0.063 | 0.849 ± 0.118 | 0.900 ± 0.065 | **0.871 ± 0.049** | 0.945 ± 0.027 |
| Krum | 0.514 ± 0.005 | 0.484 ± 0.056 | 0.107 ± 0.000 | 0.458 ± 0.087 | 0.957 ± 0.008 | 0.816 ± 0.040 | 0.143 ± 0.043 | 0.735 ± 0.046 |
| Median | 0.514 ± 0.004 | 0.476 ± 0.065 | 0.275 ± 0.071 | 0.508 ± 0.103 | 0.957 ± 0.007 | 0.864 ± 0.031 | 0.535 ± 0.106 | 0.846 ± 0.057 |
| Trimmed Mean | 0.514 ± 0.003 | 0.511 ± 0.011 | 0.230 ± 0.095 | 0.492 ± 0.073 | 0.956 ± 0.007 | 0.885 ± 0.034 | 0.570 ± 0.097 | 0.849 ± 0.047 |
| Centralized, all 5 seeds | 0.519 ± 0.009 | | | | 0.938 ± 0.075 | | | |
| Centralized, stable seeds | 0.522 ± 0.011 (seeds 0, 1, 3) | | | | 0.972 ± 0.001 (seeds 0–3) | | | |

- **Random protocol.** FedAdam is best in P0, P1 and P4 and second under P3. Under IID P0 every
  rule except SCAFFOLD is within 0.016 of the others (0.951–0.967): with honest, IID controllers
  the aggregation rule hardly matters. Under skew the robust rules fall far behind (Krum 0.143
  under P3). P3 is the hardest partition for every rule; there **SCAFFOLD (0.871), FedAdam
  (0.842) and FedNova (0.822)** clearly beat FedAvg / FedProx (0.778 / 0.771). SCAFFOLD's low and
  variable P0 / P1 values come from diverged runs (Section 9.9).
- Against the stable centralized baseline (0.972), FedAdam is within 0.001–0.005 under P0 / P1 /
  P4; FedAvg is 0.016–0.034 below.
- **Chronological protocol.** Almost everything sits at 0.51–0.53, the level of the centralized
  model, in P0, P1 and P4. Only P3 separates strategies: FedAvg and FedProx reach 0.67 (above
  centralized), the robust rules collapse (0.11–0.28). Which classes happen to transfer to the
  shifted test set matters more than the rule (Section 9.3).

Final-round and AUC values (`table_final_auc_macro_f1.csv`, random): for the averaging rules
other than SCAFFOLD, final-round scores are 0.02–0.09 below the validation-selected ones (e.g. FedAvg P0 final 0.871, AUC 0.840; FedAdam P0
final 0.924, AUC 0.840; centralized final 0.773 over 5 seeds). Validation-selected rounds average
55–93 for the averaging rules, so the runs are still improving or oscillating late in training at
these learning rates.

### 9.3 Per-class test recall at the validation-selected round

DoS / Probe / DDoS / BFA (Normal recall is 0.97–1.00 everywhere except Krum on chrono P4, 0.87):

| strategy | chrono P0 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|
| FedAvg | 0.73/0.30/1.00/0.00 | 0.81/0.62/0.99/0.00 | 0.78/0.30/1.00/0.00 | 0.99/0.97/1.00/0.79 | 0.99/0.95/0.99/0.75 | 0.98/0.97/0.95/**0.00** | 0.96/0.98/1.00/0.83 |
| FedProx | 0.73/0.30/1.00/0.00 | 0.82/0.61/0.99/0.00 | 0.77/0.30/1.00/0.00 | 0.99/0.96/1.00/0.82 | 0.96/0.97/1.00/0.76 | 0.96/0.96/0.95/**0.00** | 0.97/0.99/1.00/0.82 |
| FedNova | 0.74/0.30/1.00/0.00 | 0.81/0.38/1.00/0.00 | 0.76/0.30/1.00/0.00 | 0.99/0.97/1.00/0.80 | 0.98/0.95/1.00/0.75 | 0.88/0.93/0.99/0.34 | 0.98/0.95/1.00/0.82 |
| FedAdam | 0.74/0.30/1.00/0.00 | 0.71/0.40/1.00/0.00 | 0.79/0.30/1.00/0.00 | 0.99/0.99/1.00/0.83 | 0.99/0.99/1.00/0.84 | 0.93/0.97/0.99/0.29 | 0.99/0.99/1.00/0.84 |
| SCAFFOLD | 0.77/0.40/0.80/0.00 | 0.62/0.37/1.00/0.00 | 0.60/0.30/1.00/0.00 | 0.88/0.94/0.98/0.57 | 0.97/0.95/0.99/0.67 | 0.95/0.95/0.98/**0.38** | 0.98/0.97/1.00/0.81 |
| Krum | 0.75/0.30/1.00/0.00 | 0.00/0.00/0.00/0.00 | 0.65/0.44/0.80/0.00 | 0.99/0.98/1.00/0.80 | 0.85/0.86/0.98/0.72 | 0.00/0.00/0.60/0.00 | 0.75/0.83/0.78/0.66 |
| Median | 0.74/0.30/1.00/0.00 | 0.62/0.06/0.00/0.12 | 0.62/0.37/1.00/0.00 | 0.99/0.98/1.00/0.79 | 0.93/0.90/1.00/0.82 | 0.25/0.69/0.79/0.46 | 0.87/0.87/0.99/0.79 |
| Trimmed Mean | 0.74/0.30/1.00/0.00 | 0.17/0.13/0.00/0.35 | 0.61/0.33/1.00/0.00 | 0.99/0.97/1.00/0.79 | 0.96/0.91/0.99/0.73 | 0.29/0.79/0.83/0.43 | 0.89/0.87/0.99/0.81 |
| Centralized | 0.77/0.31/1.00/0.01 | | | 0.97/0.95/1.00/0.84 | | | |

(Chrono P1 behaves like chrono P0 for the averaging rules; full values in
`table_per_class_recall.csv`.)

- On the random protocol **BFA, the class with the fewest training rows, degrades first**: FedAvg
  BFA recall 0.79 (P0) → 0.75 (P1) → **0.00 (P3)**, while DoS, Probe and DDoS stay ≥ 0.95. Under P3
  BFA is held by one or two controllers, and plain averaging erases it; SCAFFOLD (0.38), FedNova
  (0.34) and FedAdam (0.29) keep part of it, which is most of their P3 advantage.
- Under P3, Krum follows one controller's update and loses DoS and Probe entirely; Median and
  Trimmed Mean lose much of DoS.
- On the chronological protocol BFA is 0 for nearly every rule and Probe stays near 0.30, the shift
  of Section 9.1.

### 9.4 T95 and bytes

Cells: mean T95 rounds over the seeds that reached it (seeds reaching / 5). MB to T95 = rounds ×
1.11 MB (× 2.23 MB for SCAFFOLD). Full table: `table_t95_bytes.csv`.

Random protocol, R = 100:

| strategy | P0 (all 5 / stable 4) | P1 (all / stable) | P3 | P4 (all / stable) |
|---|---|---|---|---|
| FedAvg | 26.2 (5/5) / 42.2 (5/5) | 29.8 (5/5) / 41.5 (4/5) | never | 18.0 (4/5) / 33.5 (4/5) |
| FedProx | 36.2 (5/5) / 57.4 (5/5) | 28.2 (5/5) / 65.0 (3/5) | never | 32.5 (4/5) / 40.8 (4/5) |
| FedNova | 27.4 (5/5) / 50.4 (5/5) | 43.8 (5/5) / 55.8 (5/5) | never | 39.4 (5/5) / 60.0 (4/5) |
| FedAdam | 38.0 (5/5) / 57.4 (5/5) | 29.4 (5/5) / 38.4 (5/5) | never | 25.4 (5/5) / **31.8 (5/5)** |
| SCAFFOLD | 33.3 (3/5) / 77.0 (2/5) | 41.5 (4/5) / 66.5 (2/5) | 90.0 (2/5) / 82.0 (1/5) | 33.6 (5/5) / 37.2 (4/5) |
| Krum | 33.8 (5/5) / 66.4 (5/5) | never | never | never |
| Median | 23.2 (5/5) / 43.0 (5/5) | 31.0 (1/5) / never | never | 74.5 (2/5) / never |
| Trimmed Mean | 21.0 (5/5) / 37.4 (5/5) | 59.0 (2/5) / 62.0 (1/5) | never | 70.0 (1/5) / never |

- The reference matters: with the stable threshold (0.923 instead of 0.891) T95 rises by 10–30
  rounds and several settings drop out. Conclusions that hold under both: only **FedAdam reaches
  T95 in every seed of P0, P1 and P4** under the stable threshold; no rule except SCAFFOLD (2 of 5
  seeds) reaches it under P3; the robust rules reach it only under P0.
- In MB, FedAdam needs 35–64 MB to T95 (stable threshold) outside P3; SCAFFOLD's 2.23 MB per round
  doubles its cost (e.g. P4 37.2 rounds = 83 MB).
- Chronological protocol: the threshold is low (0.493 / 0.496), and most settings cross it within
  1–13 rounds under both references (Krum / Median / Trimmed Mean never under P3). These values
  say little about convergence.

### 9.5 Random protocol: R = 30 vs R = 100

R = 30 values are the first 30 rounds of the R = 100 runs, re-selected by validation.

| strategy | P0 R30 → R100 | P1 R30 → R100 | P3 R30 → R100 | P4 R30 → R100 | T95 seeds reached at R = 30 (of 20) / R = 100 |
|---|---|---|---|---|---|
| FedAvg | 0.900 → 0.956 | 0.903 → 0.938 | 0.759 → 0.778 | 0.896 → 0.941 | 10 / 14 |
| FedProx | 0.885 → 0.957 | 0.900 → 0.929 | 0.748 → 0.771 | 0.885 → 0.946 | 6 / 14 |
| FedNova | 0.899 → 0.951 | 0.893 → 0.943 | 0.757 → 0.822 | 0.892 → 0.938 | 9 / 15 |
| FedAdam | 0.873 → 0.967 | 0.897 → 0.971 | 0.771 → 0.842 | 0.913 → 0.969 | 8 / 15 |
| SCAFFOLD | 0.827 → 0.849 | 0.871 → 0.900 | 0.825 → 0.871 | 0.892 → 0.945 | 7 / 14 |
| Krum | 0.885 → 0.957 | 0.758 → 0.816 | 0.143 → 0.143 | 0.632 → 0.735 | 1 / 5 |
| Median | 0.908 → 0.957 | 0.824 → 0.864 | 0.502 → 0.535 | 0.775 → 0.846 | 4 / 8 |
| Trimmed Mean | 0.904 → 0.956 | 0.843 → 0.885 | 0.530 → 0.570 | 0.789 → 0.849 | 5 / 8 |

(T95 counts use the all-5-seeds threshold 0.891.) Going from 30 to 100 rounds raises every
setting except Krum P3. At R = 30 the averaging rules are close together (FedAdam is not yet
ahead under P0); FedAdam's lead and FedNova / SCAFFOLD's P3 advantage appear only with more
rounds. Tuning at R = 30 therefore saw a different ordering than the R = 100 evaluation.

### 9.6 Worst-client validation macro-F1 (validation-selected round)

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.831 ± 0.008 | 0.803 ± 0.022 | 0.497 ± 0.002 | **0.826 ± 0.051** | 0.921 ± 0.008 | 0.776 ± 0.047 | 0.497 ± 0.001 | 0.749 ± 0.072 |
| FedProx | 0.826 ± 0.008 | 0.799 ± 0.023 | 0.496 ± 0.002 | 0.800 ± 0.042 | 0.924 ± 0.005 | 0.766 ± 0.044 | 0.497 ± 0.002 | 0.768 ± 0.078 |
| FedNova | 0.832 ± 0.006 | 0.745 ± 0.038 | 0.558 ± 0.135 | 0.756 ± 0.115 | 0.915 ± 0.011 | 0.750 ± 0.091 | 0.728 ± 0.069 | 0.712 ± 0.096 |
| FedAdam | **0.903 ± 0.030** | **0.807 ± 0.050** | 0.572 ± 0.109 | 0.804 ± 0.116 | **0.942 ± 0.010** | **0.827 ± 0.065** | 0.719 ± 0.051 | **0.793 ± 0.098** |
| SCAFFOLD | 0.722 ± 0.057 | 0.733 ± 0.039 | **0.757 ± 0.110** | 0.749 ± 0.077 | 0.818 ± 0.111 | 0.712 ± 0.132 | **0.745 ± 0.091** | 0.758 ± 0.099 |
| Krum | 0.888 ± 0.041 | 0.665 ± 0.090 | 0.121 ± 0.030 | 0.509 ± 0.134 | 0.926 ± 0.008 | 0.664 ± 0.109 | 0.167 ± 0.052 | 0.553 ± 0.049 |
| Median | 0.833 ± 0.003 | 0.714 ± 0.047 | 0.377 ± 0.092 | 0.716 ± 0.096 | 0.921 ± 0.010 | 0.705 ± 0.109 | 0.351 ± 0.125 | 0.659 ± 0.090 |
| Trimmed Mean | 0.834 ± 0.005 | 0.735 ± 0.060 | 0.371 ± 0.095 | 0.755 ± 0.086 | 0.917 ± 0.008 | 0.730 ± 0.076 | 0.373 ± 0.156 | 0.674 ± 0.075 |

Under P3 the worst controller is at 0.50 for FedAvg / FedProx on both protocols: it is the BFA
holder whose BFA rows are no longer recognized. SCAFFOLD (0.75), FedNova (0.73) and FedAdam (0.72)
lift the worst controller substantially. Elsewhere FedAdam gives the best worst-client score.
Worst-client scores use each client's own validation set, so they are in-distribution and are not
affected by the chronological shift.

### 9.7 Local-only baseline (random, 100 epochs per client, global test set)

| partition | local mean client | local worst | local best | classes per client | FedAvg | FedAdam | centralized (all 5 / stable) |
|---|---|---|---|---|---|---|---|
| P0 | 0.950 ± 0.007 | 0.924 ± 0.020 | 0.962 ± 0.002 | 5.0 | 0.956 ± 0.004 | 0.967 ± 0.003 | 0.938 / 0.972 |
| P1 α0.5 | 0.804 ± 0.078 | 0.495 ± 0.294 | 0.952 ± 0.006 | 4.95 | 0.938 ± 0.017 | 0.971 ± 0.002 | |
| P3 m1 | 0.214 ± 0.011 | 0.097 ± 0.000 | 0.320 ± 0.004 | 1.8 | 0.778 ± 0.005 | 0.842 ± 0.026 | |
| P4 | 0.745 ± 0.045 | 0.449 ± 0.120 | 0.906 ± 0.057 | 4.8 | 0.941 ± 0.046 | 0.969 ± 0.004 | |

- Under IID data a single controller's 6,300 rows already give 0.950, so federation adds little
  (+0.006 FedAvg, +0.017 FedAdam).
- Under skew federation is essential: under P3 a controller sees on average 1.8 of the 5 classes
  and the best local model reaches 0.320; the worst one (a Normal-only controller) predicts
  everything as Normal (0.097). FedAvg lifts every controller to 0.778 and FedAdam to 0.842.
- Under P1 / P4 the typical controller is 0.13–0.20 below FedAvg, and the worst one 0.44–0.49
  below.

### 9.8 Heterogeneity sweep (random, R = 100, FedAvg and FedAdam)

| partition | mean JSD | mean CV | FedAvg | FedAdam | FedAvg final round |
|---|---|---|---|---|---|
| P1 α = 0.1 | 0.688 | 0.979 | 0.923 ± 0.024 | 0.956 ± 0.006 | 0.875 |
| P1 α = 0.5 | 0.368 | 0.546 | 0.938 ± 0.017 | 0.971 ± 0.002 | 0.882 |
| P1 α = 1 | 0.196 | 0.457 | 0.952 ± 0.007 | 0.970 ± 0.002 | 0.858 |
| P1 α = 10 | 0.019 | 0.170 | 0.934 ± 0.012 | 0.967 ± 0.005 | 0.846 |
| P0 (IID) | 0.000 | 0.000 | 0.956 ± 0.004 | 0.967 ± 0.003 | 0.871 |
| P2 β = 0.5 | 0.002 | 1.135 | 0.943 ± 0.040 | 0.972 ± 0.001 | 0.547 |

- Dirichlet label skew costs FedAvg little on this dataset until α = 0.1 (−0.033 vs P0), and the
  trend is not monotone (α = 10 is below α = 1). The differences between α = 10, 1 and 0.5 are
  within about one standard deviation.
- FedAdam is flat (0.956–0.972) and above FedAvg at every level of skew, by 0.011–0.033.
- Pure quantity skew (P2) leaves the validation-selected score intact but makes FedAvg unstable
  late in training: its final-round mean is 0.547 because 2 of 5 runs collapsed to one class by
  round 100 (Section 9.9). FedAdam is stable there.
- Compare P3 (JSD about 0.5, attack-type exclusivity): FedAvg 0.778. Exclusivity of whole attack
  families hurts far more than Dirichlet skew of similar or higher JSD (α = 0.1, JSD 0.69, still
  0.923), because under P1 every family keeps several holders.

### 9.9 Divergence and collapse

`table_divergence_collapse.csv` (all suites). Diverged = non-finite weights; collapsed = predicts
one class for (nearly) all test flows at the final round (includes diverged runs).

| suite / protocol | strategy | runs | diverged | collapsed (final) | val-selected F1 of collapsed runs | of the others |
|---|---|---|---|---|---|---|
| random, R = 100 | **SCAFFOLD** | 20 | **5** (P0 s1, s3, s4; P1 s3; P4 s3) | 5 | 0.817 | 0.916 |
| random, R = 100 | **centralized** | 5 | 0 | **1** (seed 4) | 0.804 | 0.972 |
| random, R = 100 | Krum | 20 | 0 | 2 (P3 s1, s2) | 0.097 | 0.726 |
| chrono, R = 30 | **SCAFFOLD** | 20 | **9** (P0 all 5; P1 s0, s3; P4 s2, s4) | 9 | 0.505 | 0.510 |
| chrono, R = 30 | **centralized** | 5 | 0 | **2** (seeds 2, 4) | 0.516 | 0.522 |
| chrono, R = 30 | FedAvg | 20 | 0 | 1 (P4 s4) | 0.520 | 0.557 |
| chrono, R = 30 | Krum | 20 | 0 | 5 (P3, all seeds) | 0.107 | 0.486 |
| chrono poisoning | Krum, targeted | 5 | 0 | 3 | 0.127 | 0.351 |
| sweep, R = 100 | FedAvg | 20 | 0 | 2 (P2 β0.5 s1, s4) | 0.915 | 0.941 |
| random poisoning (P3) | FedAvg, sign flip | 5 | 2 | 3 | 0.371 | 0.688 |
| random poisoning (P3) | Krum, sign flip / targeted | 5 / 5 | 0 | 1 / 5 | 0.097 | — |
| lr sensitivity (lr 0.6) | FedAvg / FedProx | 20 / 20 | 2 / 2 (all P1) | 3 / 2 | | |
| tuning | FedAdam / Krum / Median / TM / SCAFFOLD | 36 each | 0 / 0 / 0 / 0 / 1 | 3 / 2 / 2 / 2 / 1 | | |

**SCAFFOLD divergence rate: 25% on the random protocol (5/20) and 45% on the chronological
protocol (9/20)**, all at the tuned client lr 0.2018, which was chosen on the tuning seed where the
next larger lr already diverged. On chrono P0 it diverged in every seed. Validation-based round
selection recovers a pre-divergence round, so SCAFFOLD's headline values are still 0.72–0.96 in
diverged random runs, but its final-round and AUC values are low (random P0 final 0.414) and its
mean / variance in the main table are dominated by when it diverged.

**Centralized collapse.** Random seed 4 (lr 0.2018, 100 epochs) peaked at epoch 3 (validation
0.804) and then fell to predicting one class (final test F1 0.097). Chronological seeds 2 and 4
also ended in one-class predictions, but their best validation epochs (test 0.519 / 0.512) are in
line with the stable seeds (0.522), so only the random baseline is materially affected:
**0.938 ± 0.075 over all 5 seeds vs 0.972 ± 0.001 over the 4 stable seeds.** This changes the
random T95 threshold (0.891 vs 0.923) and the per-seed gap used in the regression; both are
reported both ways (Sections 9.4 and 10).

Other collapses are either late instability of FedAvg at lr 0.256 (P2 sweep; one chrono P4 run)
or failure modes of the method itself (Krum under P3 follows a controller that holds Normal plus
one family, and predicts one class from the start), not numerical divergence.

### 9.10 Learning-rate boundary check (tuning seed 99, validation only, R = 30)

Requested after tuning, because every plain-SGD rule chose a learning rate near the top of the
search range. FedAvg, FedNova, FedProx and SCAFFOLD were run at client lr 0.4 and 0.6 (other values
as tuned; FedProx keeps μ = 0.00117) on the three tuning partitions, scored exactly like tuning.
**The frozen values were not changed and nothing else was rerun.** `table_lr_boundary.csv`.

| strategy | frozen best (lr) | lr 0.4 | lr 0.6 | better than frozen? |
|---|---|---|---|---|
| FedAvg | 0.848 (0.256) | 0.861 | 0.862 | yes (+0.014) |
| FedNova | 0.843 (0.256) | 0.845 | 0.846 | yes, marginally (+0.003) |
| FedProx | 0.842 (0.202) | 0.859 | 0.876 | yes (+0.034) |
| SCAFFOLD | 0.880 (0.202) | 0.885, **1 run diverged** | 0.844, **1 run diverged** | no (ineligible under the tuning rule) |

The search range [1e-3, 3e-1] clipped the optimum of FedAvg, FedNova and FedProx: their tuning
score keeps rising past 0.3. Their best rounds were at rounds 21–30 of 30, so a short tuning
horizon favours large steps. SCAFFOLD is already at its stability limit at 0.2.

### 9.11 Learning-rate sensitivity (random, R = 100; sensitivity only, not the headline)

FedAvg and FedProx rerun on the full random grid (4 partitions × 5 seeds) at client lr 0.6,
everything else frozen (FedProx keeps μ = 0.00117). `table_lr_sensitivity.csv`,
`stats_lr_sensitivity.csv` (Wilcoxon with Holm within each family of 2 comparisons; bootstrap CIs
as in Section 10).

| strategy / partition | frozen lr | lr 0.6 | lr 0.6 final round | lr 0.6 diverged / collapsed | FedAdam (frozen) |
|---|---|---|---|---|---|
| FedAvg P0 | 0.956 ± 0.004 | 0.921 ± 0.026 | 0.700 | 0 / 1 | 0.967 |
| FedAvg P1 | 0.938 ± 0.017 | 0.874 ± 0.065 | 0.555 | 2 / 2 | 0.971 |
| FedAvg P3 | 0.778 ± 0.005 | **0.859 ± 0.017** | 0.806 | 0 / 0 | 0.842 |
| FedAvg P4 | 0.941 ± 0.046 | 0.935 ± 0.045 | 0.885 | 0 / 0 | 0.969 |
| FedProx P0 | 0.957 ± 0.004 | 0.933 ± 0.019 | 0.854 | 0 / 0 | 0.967 |
| FedProx P1 | 0.929 ± 0.016 | 0.901 ± 0.033 | 0.555 | 2 / 2 | 0.971 |
| FedProx P3 | 0.771 ± 0.005 | **0.853 ± 0.016** | 0.820 | 0 / 0 | 0.842 |
| FedProx P4 | 0.946 ± 0.045 | 0.941 ± 0.039 | 0.873 | 0 / 0 | 0.969 |

| comparison (20 blocks) | better / worse | mean diff | 95% CI | p Holm |
|---|---|---|---|---|
| **FedAdam vs FedAvg at lr 0.6** | 17 / 3 | +0.040 | [+0.024, +0.059] | 0.005 |
| **FedAdam vs FedProx at lr 0.6** | 17 / 3 | +0.030 | [+0.019, +0.043] | 0.005 |
| FedAvg lr 0.6 vs frozen lr 0.256 | 5 / 15 | −0.006 | [−0.022, +0.006] | 0.861 |
| FedProx lr 0.6 vs frozen lr 0.202 | 8 / 12 | +0.006 | [−0.003, +0.015] | 0.985 |

- **FedAdam's advantage survives the larger learning rate**: it still beats FedAvg and FedProx at
  lr 0.6 in 17 of 20 blocks, by +0.040 / +0.030 (CIs exclude zero). The 3 losses are P3 blocks.
- **lr 0.6 is not better than the frozen lr on the R = 100 grid** (FedAvg −0.006, FedProx +0.006,
  both n.s.), although it scored higher in the R = 30 boundary check. It **trades IID / mild skew
  for P3**: P3 improves by about +0.08 (0.778 → 0.859, now above FedAdam's 0.842), while P0 and P1
  drop by 0.02–0.06, and 4 of 40 runs diverged (all P1) and 5 collapsed by the final round.
- So the clipped search range did not hide a better general setting for FedAvg / FedProx, but the
  P3 comparison is learning-rate dependent: with a larger lr, plain averaging matches or beats
  FedAdam under attack-type exclusivity (and gets closer to SCAFFOLD's 0.871).


---

## 10. Statistics

All tests use **validation-selected test macro-F1 on the random protocol at R = 100**, with
**blocks = partition × seed** (4 × 5 = 20 blocks) and 8 strategies, frozen tuned values. The
chronological protocol was not tested statistically (Section 9.2). Blocks that share a partition
scheme (same scheme, different seeds) are not fully independent, so p-values are approximate; this
caveat is written into the CSVs.

- **Friedman test:** χ² = 64.83, df = 7, **p = 1.62 × 10⁻¹¹**. The strategies do not all perform
  the same.
- **Mean ranks** (1 = best): FedAdam 1.40, FedAvg 3.73, FedNova 3.92, SCAFFOLD 4.25, FedProx 4.30,
  Trimmed Mean 5.55, Median 6.05, Krum 6.80.
- **Nemenyi post-hoc**, CD at α = 0.05 (q = 3.031 for k = 8): **CD = 2.348** ranks. Significant
  pairs (p < 0.05): FedAdam vs FedNova (0.025), SCAFFOLD (0.006), FedProx (0.004), Trimmed Mean
  (2.3 × 10⁻⁶), Median (5.4 × 10⁻⁸), Krum (8.8 × 10⁻¹¹); Krum vs FedAvg (0.002), FedNova (0.005),
  SCAFFOLD (0.022), FedProx (0.027). Cliques of not-significantly-different strategies:
  {FedAdam, FedAvg}, {FedAvg, FedNova, SCAFFOLD, FedProx, Trimmed Mean, Median},
  {Trimmed Mean, Median, Krum} (`fig6_cd_diagram.png`).
- **Planned Wilcoxon signed-rank tests vs FedAvg** (two-sided, zero differences dropped, Holm over
  7 comparisons; effect sizes: median difference and matched-pairs rank-biserial
  r = (W⁺ − W⁻)/(W⁺ + W⁻)), with **bootstrap 95% CIs** of the mean difference (seeds resampled
  within each partition, 10,000 resamples, percentile):

  | strategy | better / worse (of 20) | median diff | mean diff | 95% CI | r | p raw | p Holm |
  |---|---|---|---|---|---|---|---|
  | FedProx | 6 / 14 | −0.001 | −0.003 | [−0.005, −0.000] | −0.36 | 0.165 | 0.495 |
  | FedNova | 9 / 10 (1 tie) | −0.000 | +0.010 | [−0.000, +0.023] | +0.18 | 0.494 | 0.988 |
  | **FedAdam** | **20 / 0** | +0.022 | **+0.034** | **[+0.024, +0.046]** | +1.00 | 1.9 × 10⁻⁶ | **1.3 × 10⁻⁵** |
  | SCAFFOLD | 9 / 11 | −0.001 | −0.012 | [−0.042, +0.016] | −0.07 | 0.812 | 0.988 |
  | **Krum** | 3 / 17 | −0.156 | **−0.240** | **[−0.259, −0.220]** | −0.91 | 6.3 × 10⁻⁵ | **3.8 × 10⁻⁴** |
  | **Median** | 4 / 16 | −0.082 | **−0.103** | **[−0.129, −0.077]** | −0.86 | 2.6 × 10⁻⁴ | **1.1 × 10⁻³** |
  | **Trimmed Mean** | 3 / 17 | −0.051 | **−0.088** | **[−0.111, −0.063]** | −0.88 | 1.7 × 10⁻⁴ | **8.4 × 10⁻⁴** |

  FedProx's CI just excludes zero (−0.003), but the Wilcoxon test is not significant; the effect
  is negligible either way. The CIs ignore that seeds within a partition share the partition
  scheme only through resampling within partitions (`fig11_bootstrap_ci_vs_fedavg.png`).
- **Regression of the gap to centralized on realized heterogeneity.** Per strategy, OLS
  gap = b0 + b1·JSD + b2·CV, where gap = centralized validation-selected test macro-F1 of the same
  seed − the run's, and JSD / CV are the run's partition's realized mean pairwise JSD and shard-size
  CV. Data: the main grid (20 runs per strategy) plus the α sweep for FedAvg and FedAdam (40 runs).
  Reported twice, because random centralized seed 4 collapsed (Section 9.9): with all 5 seeds, and
  with the 4 stable seeds (runs of seed 4 dropped, so every gap uses a stable same-seed baseline).

  | strategy | n (all / stable) | JSD coef. (all) | p | JSD coef. (stable) | p | CV coef. (all / stable) | p (all / stable) | R² (all / stable) |
  |---|---|---|---|---|---|---|---|---|
  | FedAvg | 40 / 32 | 0.123 | 0.037 | 0.122 | 0.011 | −0.015 / −0.020 | 0.65 / 0.44 | 0.12 / 0.21 |
  | FedProx | 20 / 16 | 0.457 | 0.004 | 0.408 | 0.003 | −0.100 / −0.085 | 0.12 / 0.11 | 0.42 / 0.53 |
  | FedNova | 20 / 16 | 0.329 | 0.044 | 0.265 | 0.005 | −0.077 / −0.045 | 0.27 / 0.23 | 0.23 / 0.50 |
  | FedAdam | 40 / 32 | 0.092 | 0.124 | 0.073 | 0.026 | −0.018 / −0.005 | 0.60 / 0.77 | 0.06 / 0.18 |
  | SCAFFOLD | 20 / 16 | 0.143 | 0.354 | 0.148 | 0.273 | −0.128 / −0.097 | 0.07 / 0.12 | 0.19 / 0.18 |
  | Krum | 20 / 16 | 1.437 | 0.003 | 1.284 | 0.010 | −0.092 / −0.014 | 0.62 / 0.94 | 0.56 / 0.56 |
  | Median | 20 / 16 | 0.823 | 0.004 | 0.723 | 0.008 | −0.087 / −0.045 | 0.43 / 0.66 | 0.50 / 0.55 |
  | Trimmed Mean | 20 / 16 | 0.713 | 0.006 | 0.608 | 0.012 | −0.058 / −0.015 | 0.58 / 0.88 | 0.48 / 0.54 |

  - **Label skew (JSD) predicts the gap; quantity skew (CV) does not.** The JSD coefficient is
    positive for every strategy and significant (p < 0.05) for 6 of 8 with all seeds (FedAdam and
    SCAFFOLD not) and 7 of 8 with stable seeds (SCAFFOLD not). No CV coefficient is significant.
  - Sensitivity to label skew orders the strategies as the main results do: per 0.1 of JSD the gap
    grows by about 0.01 for FedAvg / FedAdam, 0.03–0.05 for FedNova / FedProx, 0.06–0.08 for
    Median / Trimmed Mean and 0.13–0.14 for Krum. (FedAvg's and FedAdam's slopes include the α
    sweep, where P1 skew costs little; the other slopes come from P0 / P1 / P3 / P4 only and are
    driven by P3.)
  - Dropping the collapsed centralized seed changes the JSD slopes by at most 0.11 (0.15 for
    Krum) and makes them more precise; the conclusions are the same.
  - JSD and CV are correlated in these designs (r = 0.45 with the sweep, 0.73 without), and P3
    contributes the largest JSD values, so the JSD slope partly measures "P3 vs the rest".
    `stats_regression_gap.csv`, `stats_regression_points.csv`, `fig10_gap_regression.png`.

**In plain words.**

- FedAdam beat FedAvg in all 20 settings, by 0.034 on average (95% CI 0.024–0.046), and it is the
  only strategy separated from the middle group by the conservative all-pairs Nemenyi test.
- FedProx (tuned μ ≈ 0.001), FedNova and SCAFFOLD are not distinguishable from FedAvg overall.
  FedNova and SCAFFOLD help under P3 and lose elsewhere (SCAFFOLD through divergence), and the two
  effects cancel in the overall comparison.
- All three robust rules are clearly worse than FedAvg when no one attacks: by about 0.09–0.10 for
  Median and Trimmed Mean and 0.24 for Krum.
- How far a strategy falls below centralized training is explained by label skew, not by shard
  size imbalance.

---

## 11. Hypotheses

Verdicts rest on the random protocol (R = 100), where differences between strategies are visible.
The chronological protocol is reported but is dominated by shift (Section 9.1).

### H1: The FedAvg–centralized gap widens as α falls, is largest under P3 and P4, and rare-class recall degrades first

**Verdict: partly supported.**

- Gap of FedAvg to the stable centralized baseline (0.972): P0 0.016, P1 0.034, **P3 0.194**,
  P4 0.031. The gap is largest under P3, as predicted, but **P4 is not harder than P1**.
- α sweep: FedAvg 0.956 (P0) → 0.934 (α 10) → 0.952 (α 1) → 0.938 (α 0.5) → 0.923 (α 0.1). The gap
  widens at the most extreme skew but is **not monotone** in α, and the middle values are within
  noise. In the regression, FedAvg's gap grows with realized JSD (p = 0.037; 0.011 with stable
  seeds), but slowly (0.012 per 0.1 JSD).
- Rare-class recall degrades first: BFA (fewest training rows) 0.79 → 0.75 → **0.00** from P0 → P1
  → P3, while the other attack classes stay ≥ 0.95. Supported.
- Quantity skew alone (P2 β = 0.5) does not lower the validation-selected score (0.943) but makes
  late training unstable (2 of 5 FedAvg runs collapsed by round 100).

### H2: FedProx, SCAFFOLD and FedAdam recover part of the gap under P3/P4 but give little benefit under IID or mild skew; FedNova's effect needs unequal τ and vanishes under P0

**Verdict: partly supported — for FedAdam, SCAFFOLD and FedNova under P3; not for FedProx; not
under P4.**

- **FedAdam** gains over FedAvg: P0 +0.011, P1 +0.033, **P3 +0.064**, P4 +0.028. The gain is
  smallest under IID and largest under exclusivity, as predicted, and it closes most of the P4 /
  P1 gap (to within 0.001–0.003 of centralized). Under P3 it recovers about a third of the gap
  (0.842 vs 0.778, centralized 0.972). This P3 advantage depends on FedAvg's learning rate: FedAvg at
  lr 0.6 reaches 0.859 under P3 (Section 9.11), while FedAdam's advantage elsewhere holds.
- **SCAFFOLD** recovers the most under P3 (0.871, +0.093 over FedAvg; BFA recall 0.38 vs 0.00) and
  is about equal under P4 (0.945 vs 0.941), but at the tuned lr it diverges under P0 (3 of 5 seeds)
  and P1 (1 of 5), so it is **worse** under IID and mild skew. Its benefit is real but conditional
  on stability.
- **FedProx** with its tuned μ = 0.00117 recovers nothing (within 0.009 of FedAvg everywhere). The
  tuning chose a μ so small that the proximal term is inactive.
- **FedNova** under P0: nearly equal to FedAvg (0.951 vs 0.956). The P0 shards have equal size, but
  τ_k can still differ by one step after masking Rare, and at lr 0.256 small differences are
  amplified; in v1 (lr 0.05) the two were identical. Under P3, where τ_k differ most, FedNova is
  clearly better (+0.044); under P1 / P4 it is within ±0.005. Supported.

### H3: Without adversaries, Krum and Median lose accuracy relative to FedAvg under non-IID partitions; Trimmed Mean lies in between

**Verdict: supported (random protocol).**

- Under P1 / P3 / P4: Krum 0.816 / 0.143 / 0.735, Median 0.864 / 0.535 / 0.846, Trimmed Mean
  0.885 / 0.570 / 0.849, against FedAvg 0.938 / 0.778 / 0.941.
- Trimmed Mean lies between Median and FedAvg in all three skewed partitions (narrowly under P4).
- Under IID P0 all three equal FedAvg (0.956–0.957).
- All three are significantly worse than FedAvg (Wilcoxon, Holm p ≤ 0.0011; CIs exclude zero), and
  their gap to centralized grows fastest with JSD (Section 10).
- Chronological protocol: the P3 collapse is reproduced (Krum 0.107, Median 0.275, Trimmed Mean
  0.230 vs FedAvg 0.675).

### H4: With poisoning controllers, robust rules beat FedAvg under sign flipping, but targeted attack-to-benign relabeling stays hard for every rule under P3 and P4

**Verdict: first part not supported; second part supported under P3, not under chronological P4.**

Chronological P4, R = 30, f = 2 (`table_poisoning.csv`):

| strategy | clean F1 / FNR | sign flip F1 / FNR | targeted F1 / FNR |
|---|---|---|---|
| FedAvg | 0.519 ± 0.006 / 0.036 | 0.527 ± 0.017 / 0.101 | 0.485 ± 0.061 / 0.047 |
| Krum | 0.458 ± 0.087 / 0.065 | 0.349 ± 0.167 / 0.050 | 0.216 ± 0.177 / **0.581** |
| Median | 0.508 ± 0.103 / 0.060 | 0.498 ± 0.087 / 0.062 | 0.482 ± 0.080 / 0.122 |
| Trimmed Mean | 0.492 ± 0.073 / 0.035 | 0.490 ± 0.082 / 0.033 | 0.472 ± 0.073 / 0.072 |

Random P3 m = 1, R = 100, f = 2 (`table_poisoning_random_P3.csv`; "sole" = runs in which a malicious
controller is the only holder of an attack family: seeds 0 (DoS), 2 (Probe), 4 (BFA)):

| strategy | clean F1 / FNR | sign flip F1 / FNR | sign flip F1, sole / not sole | targeted F1 / FNR | targeted F1, sole / not sole |
|---|---|---|---|---|---|
| FedAvg | 0.778 ± 0.005 / 0.002 | 0.497 ± 0.298 / 0.006 | 0.371 / 0.688 | 0.675 ± 0.127 / **0.206** | 0.615 / 0.765 |
| Krum | 0.143 ± 0.043 / 0.544 | 0.158 ± 0.035 / 0.408 | 0.151 / 0.168 | 0.097 ± 0.000 / **1.000** | 0.097 / 0.097 |
| Median | 0.535 ± 0.106 / 0.098 | 0.416 ± 0.142 / 0.086 | 0.357 / 0.506 | 0.461 ± 0.149 / **0.253** | 0.378 / 0.585 |
| Trimmed Mean | 0.570 ± 0.097 / 0.098 | 0.456 ± 0.070 / 0.071 | 0.421 / 0.507 | 0.463 ± 0.064 / **0.196** | 0.442 / 0.495 |

- **Sign flip.** On chrono P4 it does not hurt FedAvg (0.527 vs 0.519 clean; one run's FNR rose to
  0.37). On random P3 it hurts FedAvg badly (−0.281 on average; 2 of 5 runs diverged), but the
  robust rules are **not better**: their attacked scores (0.16–0.46) stay below FedAvg's (0.497),
  because they already start much lower without attackers. Their relative loss is smaller
  (Median −0.119, Trimmed Mean −0.114) but they never overtake FedAvg.
- **Targeted relabeling** raises attack FNR for every rule under P3 (FedAvg 0.002 → 0.206, Median
  0.098 → 0.253, Trimmed Mean 0.098 → 0.196, Krum → 1.000), so it is hard for every rule there, as
  predicted. On chrono P4 only Krum is badly affected (FNR 0.58).
- **Sole holder.** When a malicious controller is the only holder of an attack family, every rule
  loses more: FedAvg sign flip 0.371 vs 0.688, targeted 0.615 vs 0.765; Median and Trimmed Mean
  follow the same pattern. A robust rule cannot recover a family whose only source is malicious —
  the honest information is simply absent. (3 of 5 seeds are "sole" cases, so these splits rest on
  3 vs 2 runs.)

---

## 12. Figures and tables

All under `results/`, produced by `scripts/paper_outputs.py`, `scripts/paper_diagrams.py` and
`scripts/flower_validation.py --plot-only`. Figures: 300 dpi, STIXGeneral font, ≥ 8 pt, Okabe-Ito
colors fixed per strategy (SCAFFOLD grey `#999999`, centralized black), no titles. Tables: CSV,
3 decimals, mean and std in separate columns, p-values to 3 significant digits.

### Figures (`results/paper_figures/`)

| file | suggested caption | section |
|---|---|---|
| `fig0_system_architecture.png` | System model. Eight SDN controllers (C1–C8), each training locally on the flow data of its own switch domain (E local epochs of SGD), send model updates Δw_k to an aggregation server running one of eight strategies, and receive the global model w^(t+1). The global model is evaluated on a held-out test set never seen by clients. Two controllers are marked malicious for the poisoning study. Raw traffic never leaves a controller. | Methodology |
| `fig_model_architecture.png` | Detection model: MLP with 68 inputs, hidden layers of 128 and 64 units (ReLU, dropout 0.2) and 5 output classes (Normal, BFA, DDoS, DoS, Probe); 17,413 parameters (68 KB in float32). | Implementation |
| `fig_pipeline.png` | Experimental pipeline: InSDN (3 CSVs, 343,889 flows) → preprocessing → partitioner (P0, P1 α = 0.5, P3 m = 1, P4 α = β = 0.5; K = 8; seeds 0–4) → plain-PyTorch simulation (MLP, E = 1, B = 256, plain SGD, hyperparameters tuned by a 12-trial random search on seed 99) → 8 aggregation strategies (poisoning with f = 2 of 8 on chronological P4 and random P3) → metrics and statistics. | Implementation |
| `fig1_leakage_diagnostic.png` | Leakage diagnostic: test macro-F1 and per-class recall of the centralized MLP (30 epochs, mean ± std over 5 seeds) on the chronological and the random stratified split of the same rows. The chronological split loses BFA entirely and most of Probe. (The random std includes one seed that collapsed after epoch 3.) | Results 9.1 |
| `fig2_heatmap_macro_f1.png` | Validation-selected test macro-F1 (mean ± std, 5 seeds) for 8 strategies × 4 partitions with tuned hyperparameters: (a) random split, R = 100; (b) chronological split, R = 30. Centralized values under each panel are over all 5 seeds (random: 0.972 ± 0.001 over the 4 stable seeds). Panels use different protocols and must not be compared directly. | Results 9.2 |
| `fig3_convergence_random_r100.png` | Test macro-F1 per round (mean ± std over 5 seeds), random split, R = 100, under P0 (top) and P3 (bottom). Dashed: centralized validation-selected macro-F1 (100 epochs, all 5 seeds). SCAFFOLD's P0 curve falls because 3 of its 5 runs diverged. | Results 9.2 |
| `fig4_per_class_recall_random_P3.png` | Per-class test recall at the validation-selected round under P3 (m = 1), random split, R = 100, 8 strategies (mean ± std, 5 seeds). | Results 9.3 |
| `fig5_poisoning.png` | Poisoning with f = 2 of 8 malicious controllers: test macro-F1 (top) and attack false-negative rate (bottom) for FedAvg, Krum, Median and Trimmed Mean, clean vs sign flip vs targeted (attack → Normal) relabeling; (a) chronological P4, R = 30; (b) random P3, R = 100 (mean ± std, 5 seeds). | Results 11 (H4) |
| `fig6_cd_diagram.png` | Critical-difference diagram (Nemenyi, α = 0.05, CD = 2.35) of mean ranks of the 8 strategies over 20 partition × seed blocks, random split, R = 100, validation-selected test macro-F1 (Friedman p = 1.6 × 10⁻¹¹). Strategies joined by a bar are not significantly different. | Results 10 |
| `fig7_flower_vs_loop.png` | Test macro-F1 per round of the plain PyTorch simulation and a Flower simulation for FedAvg, FedAdam and Median (v1 configuration: random split, P4, seed 0, R = 30, client lr 0.05). | Implementation (validation), Section 15 |
| `fig8_tuning.png` | Hyperparameter tuning: score (mean pooled validation macro-F1 over P1, P3, P4; seed 99; R = 30) of each of the 12 trials against the client learning rate, per strategy; stars mark the frozen choice, × a diverged (ineligible) trial. FedProx and FedAdam also vary μ / server lr and ε between trials. | Methodology / Results 5.1 |
| `fig9_heterogeneity_sweep.png` | Validation-selected test macro-F1 of FedAvg and FedAdam (random, R = 100, 5 seeds) across P1 α ∈ {0.1, 0.5, 1, 10}, P0 and P2 β = 0.5, with mean realized JSD; dashed: centralized. | Results 9.8 |
| `fig10_gap_regression.png` | OLS coefficients (95% CI) of the gap to centralized on realized JSD (left) and shard-size CV (right) per strategy; circles: all 5 centralized seeds; squares: the 4 stable seeds. Random split, R = 100; FedAvg / FedAdam include the α sweep. | Results 10 |
| `fig11_bootstrap_ci_vs_fedavg.png` | Mean difference in validation-selected test macro-F1 to FedAvg with bootstrap 95% CIs (seeds resampled within partitions, 20 blocks, random R = 100); * = significant in the Holm-corrected Wilcoxon test. | Results 10 |
| `fig12_local_only.png` | Local-only training (each controller alone, 100 epochs; bar = mean client, whisker = worst to best client) vs FedAvg and FedAdam, test macro-F1 on the global test set, random split, 5 seeds; dashed: centralized. | Results 9.7 |

### Tables (`results/paper_tables/`)

| file | suggested caption | section |
|---|---|---|
| `table_tuning_trials.csv` | All 12 trials per strategy: sampled values, tuning score, divergence, rank (test F1 recorded but not used). | 5.1 |
| `table_tuning_chosen.csv` | Frozen values per strategy with best / median / min / max / quartiles / second-best score and diverged trials. | 5.1 |
| `table_leakage_diagnostic.csv` | Centralized MLP, 30 epochs, 5 seeds: macro-F1 and per-class recall, chronological vs random split. | 9.1 |
| `table_valselected_macro_f1.csv` | Validation-selected test macro-F1 (mean, std, 5 seeds) per strategy and partition; chronological R = 30, random R = 100; centralized over all seeds and over stable seeds. | 9.2 (headline) |
| `table_final_auc_macro_f1.csv` | Final-round and AUC test macro-F1 and the mean selected round. | 9.2 |
| `table_per_class_recall.csv` | Per-class test recall at the validation-selected round. | 9.3 |
| `table_t95_bytes.csv` | Rounds and MB to T95 with seeds reaching / censored, for both centralized references (all seeds; stable seeds); bytes per round (SCAFFOLD 2×). | 9.4 |
| `table_random_r30_vs_r100.csv` | Random split: validation-selected macro-F1 at R = 30 (first 30 rounds) vs R = 100, selected rounds, seeds reaching T95. | 9.5 |
| `table_worst_client_val_f1.csv` | Worst-client validation macro-F1 at the validation-selected round. | 9.6 |
| `table_local_only.csv` | Local-only baseline: mean / worst / best client test macro-F1, classes per client, with FedAvg, FedAdam and centralized for reference. | 9.7 |
| `table_alpha_sweep.csv` | FedAvg and FedAdam across P1 α, P0 and P2 with realized JSD / CV. | 9.8 |
| `table_divergence_collapse.csv` | Per suite, protocol, strategy and attack: runs, diverged, collapsed, and the collapsed run ids. | 9.9 |
| `table_lr_boundary.csv` | Tuning-seed scores at lr 0.4 / 0.6 vs the frozen best. | 9.10 |
| `table_lr_sensitivity.csv` | FedAvg / FedProx at the frozen lr vs lr 0.6 per partition, with FedAdam for reference. | 9.11 |
| `table_poisoning.csv` | Chronological P4, R = 30, f = 2: macro-F1 and attack FNR, clean / sign flip / targeted. | 11 (H4) |
| `table_poisoning_random_P3.csv` | Random P3, R = 100, f = 2: as above, plus sole-holder runs and their families. | 11 (H4) |
| `table_partition_heterogeneity.csv` | Realized partition statistics per protocol, partition and seed (incl. sweep and tuning partitions). | 4 |
| `stats_friedman.csv`, `stats_mean_ranks.csv`, `stats_nemenyi_pvalues.csv` | Friedman test (8 strategies, 20 blocks), mean ranks, Nemenyi p-values. | 10 |
| `stats_wilcoxon_vs_fedavg.csv` | Wilcoxon vs FedAvg, Holm over 7, median difference, rank-biserial r. | 10 |
| `stats_bootstrap_ci_vs_fedavg.csv` | Bootstrap 95% CIs of the mean difference to FedAvg. | 10 |
| `stats_regression_gap.csv`, `stats_regression_points.csv` | OLS of the gap to centralized on JSD and CV per strategy, for both centralized references; and the data points. | 10 |
| `stats_lr_sensitivity.csv` | Sensitivity grid: FedAdam vs FedAvg / FedProx at lr 0.6, and lr 0.6 vs frozen lr (Wilcoxon + Holm within each family of 2, bootstrap CIs). | 9.11 |
| `flower_validation.csv` | Flower vs plain loop (v1 configuration). | 15 |

---

## 13. Deviations from the planned methodology (paper Section III)

| item | planned | done | reason |
|---|---|---|---|
| Datasets | InSDN, CICDDoS2019, CICIDS2017, UNSW-NB15 | InSDN only | Time-boxed; the paper runs InSDN first |
| Tasks | binary and multi-class | multi-class only | Time; partitions are built so the binary task can reuse them |
| Split | per class, earliest 80% train, ±60 s gap discarded | per family, cut between whole minutes, no gap; fallbacks relaxed cut / grouped pair split / all-train | The gap rule would have discarded 28–70% of most attack families and left BFA and BOTNET without test flows |
| Deduplication | "remove duplicate rows" | on features + label after identifier removal; conflicting-label groups dropped; earliest kept | Duplicates differing only in identifiers would leak across the split |
| Identifiers | drop flow IDs, IPs, timestamps | also drop `Src Port`; keep `Dst Port` | Source port is ephemeral; destination port carries service information |
| Random split | leakage diagnostic for the centralized baseline only | full secondary protocol (all strategies, R = 100); statistical tests and tuning run on it | Chronological centralized macro-F1 is about 0.52 due to shift; strategy differences are not visible there |
| Partitions | 10 settings | all 10 built (chrono); main grid P0, P1 α0.5, P3 m1, P4; sweep P1 α 0.1 / 1 / 10 and P2 β0.5 (FedAvg, FedAdam only); P2 β1, P3 m2 not run | Time |
| P4 | "Dirichlet α = 0.5 with quantity β = 0.5" | product reweighting q_{c,k}·s_k, normalized per family | Method not specified in the paper; reduces to P1 / P2 at the limits (approved) |
| Strategies | 8 incl. SCAFFOLD | 8 incl. SCAFFOLD (variates aggregated with n_k weights) | — |
| Framework | Flower simulation, each client a process; measured message sizes | plain PyTorch, clients sequential in one process; bytes derived | No Ray wheel on Windows + Python 3.13, and speed. Validated against Flower on one (v1) configuration (Section 15) |
| Tuning | 12 random-search trials per strategy (lr, μ, η_s, ε), equal budget | as planned, on the random protocol, R = 30, tuning seed 99, 3 partitions; client lrs drawn once and shared; centralized tuned the same way | The plan does not fix the protocol or horizon for tuning; R = 30 to keep it affordable. The optimum lies at the edge of the range for the plain-SGD rules (Section 9.10) |
| K, E | K ∈ {4, 8, 16}, E ∈ {1, 5} sweeps | K = 8, E = 1 only | Time |
| Rounds | R = 100 | random R = 100; chronological R = 30 | Time; chronological scores plateau within a few rounds |
| Seeds | 5, or 10 for headline configurations | 5 (0–4) everywhere; tuning seed 99 | Time |
| Baselines | centralized, local-only, FedAvg-IID; same tuning budget | all three; centralized tuned with the same 12 shared lrs; local-only uses the centralized lr | — |
| Model | MLP + 1D-CNN | MLP only | Time |
| Round selection | not specified | headline = test score at the best pooled-validation round; final and AUC also reported | Avoids an arbitrary last round; test never used for selection |
| Rare-class recall | mean recall over the lowest-quartile classes | recall of the single class with the fewest training rows (DDoS chrono, BFA random) | With 5 classes the lowest quartile is one class |
| Fairness | std and minimum of per-client F1, number of present classes | worst-client validation F1; classes per client for local-only | Not implemented further |
| Precision, CPU time, server time | reported | per-round wall time only | Not implemented; sequential simulation |
| T95 | restricted mean min(T95, R), fraction reaching | mean over seeds reaching + count censored; for two centralized references | Centralized instability (Section 9.9) |
| Poisoning | K ∈ {8, 16}, f ∈ {0, 1, 2}, P0 and P4, InSDN + CICIDS2017, three attacks | K = 8, f = 2; chronological P4 (R = 30) and random P3 (R = 100); sign flip + targeted | Time; random label flip not run; P3 added to test the sole-holder case |
| Statistics | Friedman/Nemenyi over dataset × task × partition blocks; within-configuration Wilcoxon on 10 seeds; bootstrap CIs; regression on JSD/CV | Friedman/Nemenyi, Wilcoxon + Holm, bootstrap CIs and JSD/CV regression over 20 partition × seed blocks (random, R = 100) | One dataset / task, 5 seeds |
| Emulated testbed (Mininet + ONOS) and deployment-mode timing | planned | not done | Time |
| Resumability | `ckpt.pt` per round with RNG states | per-run completion marker (JSON); interrupted runs restart from round 1 | Runs take 20–120 s |

---

## 14. Limitations, threats to validity and future work

### Limitations and threats to validity

- **Scope:** one dataset, one task, one model, K = 8, E = 1, 5 seeds, 4 main partitions. The results
  may not generalize to other datasets, feature spaces or larger federations.
- **Tuning clipped the learning-rate optimum.** Every plain-SGD rule chose a client lr of 0.20–0.26
  near the top of the range [1e-3, 3e-1], only 2 of the 12 shared draws fell above 0.04, and on the
  tuning seed FedAvg (+0.014), FedNova (+0.003) and FedProx (+0.034) score higher still at lr 0.4–0.6
  (Section 9.10). The frozen values were kept. On the full R = 100 grid lr 0.6 is not better
  overall (Section 9.11), but it changes the P3 ordering: FedAvg / FedProx at lr 0.6 reach
  0.85–0.86 under P3, above FedAdam's tuned 0.842. P3 conclusions are therefore lr-dependent.
- **Tuning setting differs from evaluation.** Tuning used R = 30 on the random protocol; the
  ordering at R = 30 differs from R = 100 (Section 9.5), so the tuned values favour fast early
  progress. Robust rules were tuned without attackers. The same tuned values were used on the
  chronological protocol.
- **Instability at the tuned learning rates.** SCAFFOLD diverged in 25% of random-protocol and 45%
  of chronological runs; one random and two chronological centralized runs collapsed to one class;
  FedAvg collapsed late in 2 of 5 P2 runs. Validation-based round selection limits the damage to
  the headline numbers, but SCAFFOLD's results reflect its instability as much as its method, the
  random centralized baseline depends on whether seed 4 is counted (0.938 vs 0.972), and T95 and
  the gap regression are reported both ways. A stability-aware tuning rule (e.g. requiring no
  divergence or collapse on additional seeds) would likely pick a smaller lr for SCAFFOLD and the
  centralized baseline.
- **Chronological test set:** BFA's test set is a single address pair, unseen in training; DoS has
  only 10.8% test and DDoS 42.7%; centralized macro-F1 is about 0.52, so strategy comparisons on
  this protocol are dominated by shift and are not reliable.
- **Random-protocol leakage:** flows from the same sessions fall on both sides of the split, so
  scores are inflated. The statistical ranking and the tuning rest on this protocol, and the ranking
  without leakage could differ.
- **Training budget:** at R = 100 many validation-selected rounds are in the 70s–90s, so the
  averaging rules were still improving or oscillating.
- **Rare classes:** BOTNET, Web-Attack and U2R (364 flows) are excluded from the multi-class task.
  Under P3 some clients hold only Normal + Rare and therefore train on Normal only.
- **Statistics:** 20 blocks that share 4 partition schemes are not independent, so p-values are
  approximate; the bootstrap resamples seeds within partitions but cannot add partition-level
  variety. The regression has 16–40 points per strategy, JSD and CV are correlated (r = 0.45–0.73),
  and P3 dominates the high-JSD end. No tests on the chronological protocol.
- **Poisoning:** f = 2 only; random P3 sole-holder comparisons rest on 3 vs 2 runs; the malicious
  set {0, 3} occurs in 2 of 5 seeds; random label flipping not run.
- **Cost measures:** bytes are derived, not measured (Flower measured 2.8% more in v1); wall time
  comes from 4 processes on 2 physical cores; client CPU and server time not measured.
- **Implementation:** a sequential plain-PyTorch simulation, validated against Flower for one v1
  configuration and three strategies (Section 15), not for SCAFFOLD or the tuned values.

### Future work

- A stability-aware tuning rule and a wider learning-rate range (or a log grid above 0.1), tuned at
  the evaluation horizon (R = 100).
- The remaining planned parts: the binary task, the other three datasets, P2 β = 1 and P3 m = 2,
  the K and E sweeps, the 1D-CNN, 10 seeds for headline configurations.
- Revisit the chronological split for InSDN (families with more address pairs or more distinct
  minutes), or report per-class results that exclude shift-limited classes.
- Poisoning with f ∈ {1, 2} under P0 and K = 16, random label flipping, and defences that do not
  rely on majority agreement (the sole-holder case).
- Measure client CPU and server time; extend the Flower validation to SCAFFOLD and the tuned
  configuration; run the Mininet / ONOS transfer check.

---

## 15. Flower validation of the plain PyTorch simulation

> **Version note.** This validation was run in **v1** (commit `6c241e8`): untuned client lr 0.05,
> FedAdam η_s = 0.01, ε = 1e-3, before SCAFFOLD existed. It still applies to v2 because the
> training loop (`local_train`) and the FedAvg / FedAdam / Median server code are unchanged in v2;
> v2 only added a SCAFFOLD gradient-correction argument (off for every other strategy), the
> divergence / collapse flags and the tuned hyperparameter values. SCAFFOLD and the tuned
> configuration were not re-validated in Flower. Unless marked, numbers below are v1 numbers, and
> "the reduced study" means v1.

### Why it was needed

The reduced study did not use Flower (Section 13, "Framework"). All v1 result runs (285) come from
`src/fedsdn/sim.py`, a plain PyTorch loop that trains the 8 clients one after another in one
process. That leaves three open questions for anyone reading the results:

1. Would a real federated framework, with separate client processes and serialized messages, give
   the same results?
2. Is our own aggregation code correct, or do the strategy rankings partly reflect bugs in it?
3. Are the derived message sizes (params × 4 B × 2K) close to what is actually sent?

This validation answers all three for a representative configuration.

### What was run

`scripts/flower_validation.py` (commit `6c241e8`, 2026-11-06 11:32), WSL venv,
`flwr[simulation] 1.43.0` with `ray 2.55.1`, `flwr.simulation.run_simulation` (the same entry
point as `scripts/smoke_flower.py`). Each client is a Ray actor with 1 CPU, so up to 4 clients
trained in parallel on the 4 logical threads.

- **Configuration:** random protocol, P4 (α = β = 0.5), seed 0, K = 8, E = 1, R = 30, batch 256,
  client SGD lr 0.05. The same partition file, model and seed as the plain-loop run
  `random_<strategy>_P4_a0.5b0.5_s0`.
- **Three strategies, chosen so that both Flower's code and ours are checked:**

  | run | server-side aggregation | what it checks |
  |---|---|---|
  | FedAvg | **Flower's built-in `FedAvg`** | our `aggregate.fedavg` against an independent implementation |
  | FedAdam | custom Flower strategy that calls **our** `aggregate.FedAdam` | our server optimizer (with state across rounds) inside Flower's message flow |
  | Median | **Flower's built-in `FedMedian`** | our `aggregate.coordinate_median` against an independent implementation (Flower takes the median of client weights; because all clients start from the same global weights, this equals our median of updates) |

- **Clients:** each Flower `ClientApp` is one controller. It loads only its own shard (by
  `partition-id` = `client_id`, validation rows excluded, Rare masked) and trains with the
  **unchanged** `sim.local_train`: same MLP, plain SGD, class-weighted loss and batch order.
- **Federated statistics round:** before round 1 the server sends a `QUERY` message to every
  client. Each client replies with its count, per-feature sum and sum of squares, and class counts.
  The server computes the global mean / std and class weights (`scaling.py`, `class_weights`) and
  sends them in the training configuration. This is the protocol of Section 5 carried out over
  real messages, with no pooled data at the server.
- **Evaluation:** the server evaluates the global model on the test set after every round
  (`evaluate_fn`); clients never see the test set.
- **Message size:** every training message sent and received was serialized with Flower's own
  protobuf encoder (`message_to_proto(...).SerializeToString()`) and its length recorded.
- **Wall time:** server time between successive rounds, excluding the evaluation itself.

### Matching the random-number streams

In the plain loop, dropout draws from one global PyTorch random stream that runs on across all
clients and rounds. In Flower, each client trains in its own process with its own stream. So
even with identical data and batch order, the two implementations draw different dropout masks.
To separate *implementation* differences from *random-stream* differences, `RunConfig` gained an
option `per_step_seed` (off by default). It re-seeds PyTorch at the start of each (round, client)
step with the same number the Flower client uses. Each strategy was therefore run three ways:

- **Flower**;
- **plain loop, matched** (`per_step_seed=True`, run for this validation);
- **plain loop, original** (the published R = 30 random-protocol run, one global stream).

The option does not change any published result.

### Results

`results/paper_tables/flower_validation.csv`; per-round values in `results/flower_validation/`.
Differences are in test macro-F1 and are given to 6 decimals because they are far below 0.001.

| | FedAvg | FedAdam | Median |
|---|---|---|---|
| final macro-F1: Flower | 0.811 | 0.825 | 0.571 |
| final macro-F1: plain loop, matched | 0.810 | 0.825 | 0.571 |
| final macro-F1: plain loop, original | 0.816 | 0.828 | 0.565 |
| **max per-round \|diff\|, Flower vs matched** | **0.001184** | **0.000055** | **0.000052** |
| mean per-round \|diff\|, Flower vs matched | 0.000049 | 0.000002 | 0.000005 |
| max per-round \|diff\|, Flower vs original | 0.022 | 0.034 | 0.007 |
| max per-round \|diff\|, matched vs original | 0.021 | 0.034 | 0.007 |
| final diff, Flower vs original | −0.005 | −0.003 | +0.005 |
| measured bytes per round: upload / download | 567,541 / 577,827 | 567,551 / 577,837 | 567,551 / 577,837 |
| measured total ÷ derived (1,114,432 B) | 1.028 | 1.028 | 1.028 |
| wall time per round: Flower / plain loop | 1.56 s / 0.64 s | 1.30 s / 0.67 s | 1.26 s / 0.57 s |

Figure: `results/paper_figures/fig7_flower_vs_loop.png` (three panels; the Flower curve and the
matched plain-loop curve lie on top of each other).

**What the numbers mean.**

- **The implementations agree.** With matched random streams, Flower and the plain loop differ by
  at most 0.0012 macro-F1 in any round. FedAdam and Median agree to about 5 × 10⁻⁵. The FedAvg
  difference is floating-point rounding: Flower averages the full weight vectors in NumPy, while
  we average updates in float32 and add them to the global weights, so the last bits differ and
  drift slightly over 30 rounds.
- **The remaining gap to the published runs is random-stream noise, not Flower.** Flower vs
  original and matched vs original differ by the same amount (up to 0.021 / 0.034 / 0.007 per
  round). The final-round values move by at most 0.005.
- **Real messages are 2.8% larger than the derived figure.** Upload is 1.9% above the derived
  557,216 B (17,413 params × 4 B × 8 clients), because of protobuf framing and array names /
  shapes. Download is 3.7% above, because it also carries the per-round configuration, including
  the 68-value mean and std vectors and the 5 class weights. The overhead is identical for all
  three strategies.
- **Flower is about 2–2.4× slower per round** than the plain loop, even with 4 clients training in
  parallel against one sequential thread. The extra time is Ray scheduling and message
  serialization, which dominate at this small model size.

### How this supports the experiment

1. **It closes the main implementation deviation** (Section 13, "Framework"). The results in
   Sections 9–11 come from the plain loop. This run shows that a real Flower deployment of the
   same configuration gives the same per-round scores to within 0.0012. Those results can
   therefore be read as what Flower would have produced, which matches the paper's plan.
2. **It independently checks our aggregation code.** Flower's built-in FedAvg and FedMedian were
   written by other people. They agree with our `fedavg` and `coordinate_median`, so the gaps
   between FedAvg and the robust rules (Sections 9.2 and 10) are not artifacts of our median code.
   FedAdam ran correctly inside Flower's message flow, keeping its m and v state across rounds,
   and it is the strategy behind the study's main positive result.
3. **It shows the privacy-preserving setup works with real messages.** Standardization and class
   weights were computed from per-client sums over actual message exchange, with the server never
   seeing raw rows. This is the "raw traffic never leaves a controller" property of Fig. 0.
4. **It sizes the noise from implementation details.** Changing only how dropout is seeded moved
   per-round scores by up to 0.034 and final scores by up to 0.005. That is far smaller than the
   effects the conclusions rest on: in v2, FedAdam vs FedAvg +0.034 on average (20 of 20 blocks)
   and robust rules vs FedAvg −0.09 to −0.24 (v1: +0.051, −0.09 to −0.26). The rankings do not
   depend on such details.
5. **It calibrates the cost figures.** The derived 1.114 MB per round understates real serialized
   traffic by 2.8%. Every MB-to-T95 value in Section 9.4 can be multiplied by 1.028 for measured
   bytes. Because the overhead is the same for every strategy, no comparison changes.
6. **It qualifies the wall-time numbers.** The plain loop's per-round times are simulation times,
   not deployment times: a framework adds roughly 2× overhead at this model size. Wall time was
   never used to rank strategies.

### Limits of the validation

- One configuration: random protocol, P4, seed 0, R = 30, v1 hyperparameters, and three of the
  seven v1 strategies (SCAFFOLD, added in v2, was not validated).
  FedProx, FedNova, Krum and Trimmed Mean, the chronological protocol and the poisoning runs were
  not repeated in Flower. FedProx and FedNova change only client-side code that Flower runs
  unchanged (`local_train`) or a server rule of the same kind as the two that were checked. Krum
  and Trimmed Mean have Flower built-ins (`Krum`, `FedTrimmedAvg`) that could be compared in the
  same way. Note that `FedTrimmedAvg` trims a fraction β rather than a count f̂.
- The comparison is on per-round test macro-F1, not on model weights directly.
- Message sizes are serialized protobuf sizes in simulation. Transport-level overhead (gRPC, TLS,
  TCP) in a real deployment is not included.
- Each configuration took about 1.5 minutes (file timestamps 11:28:49, 11:30:14 and 11:31:42 for
  the three strategies, each including its matched plain-loop run).

Reproduce (WSL venv, repo root):

```bash
PYTHONPATH=src ~/venvs/fedsdn/bin/python scripts/flower_validation.py            # full run
PYTHONPATH=src ~/venvs/fedsdn/bin/python scripts/flower_validation.py --plot-only  # table + figure only
```
