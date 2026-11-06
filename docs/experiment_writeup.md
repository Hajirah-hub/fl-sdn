# Experiment write-up: federated aggregation strategies on InSDN (reduced study)

This document records what was actually implemented and run, so the implementation, results and
discussion sections of the paper can be written from it. Every number comes from the code, the git
history, `CLAUDE.md`, `data/processed/*/meta.json`, the partition sidecars, `results/report/`,
`results/paper_tables/` or `results/paper_figures/`. Where something is not recorded, this
document says so. Numbers that were computed only for this write-up (from stored outputs, not
part of the original pipeline) are marked **[computed for this write-up]**.

Section III of `docs/paper.pdf` is called "the paper" or "the plan" below.

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
seeds, tuning, K and E sweeps, about 8,000 runs) was not run. A **time-boxed reduced study** was
run instead:

- Dataset: **InSDN only**. Task: **multi-class only** (5 classes: Normal, BFA, DDoS, DoS, Probe).
- K = 8 controllers, E = 1 local epoch, batch 256, client SGD learning rate 0.05, MLP.
- Partitions: **P0** (IID), **P1** (Dirichlet label skew, α = 0.5), **P3** (attack-type
  exclusivity, m = 1), **P4** (combined label + quantity skew, α = β = 0.5).
- **7 strategies**: FedAvg, FedProx, FedNova, FedAdam, Krum, Median, Trimmed Mean (no SCAFFOLD).
- Seeds 0, 1, 2.
- **Two evaluation protocols**:
  - **chronological** (primary, as in the paper): R = 30 rounds;
  - **random** stratified 80/20 split of the same rows (secondary): R = 100 rounds.
- Centralized baseline per protocol, trained for as many epochs as the protocol has rounds.
- Poisoning: chronological protocol, P4, f = 2 of 8 malicious controllers, sign flip and targeted
  (attack → Normal) relabeling, against FedAvg, Krum, Median and Trimmed Mean.
- Simulation in **plain PyTorch** (all clients run one after another in one process), not
  Flower. A Flower validation run is in progress and is not part of this document.

Final results: 111 chronological runs (R = 30) and 87 random-protocol runs (R = 100). The
headline metric is **validation-selected test macro-F1**: the test score at the round with the
best pooled validation macro-F1.

### Main findings

1. **The chronological split is dominated by distribution shift.** The centralized model reached
   only 0.521 test macro-F1 on the chronological split, against 0.924 (30 epochs) and 0.971
   (100 epochs) on the random split of the same rows. BFA test recall was 0 on the chronological
   split, because its test set is a single held-out address pair, and Probe recall was about 0.30.
   On the chronological protocol, differences between strategies are small next to this gap and
   are not consistent across partitions.
2. **FedAdam was the best strategy on the random protocol (R = 100).** It had the best score in
   all 12 partition × seed blocks (mean rank 1.0), was +0.051 macro-F1 above FedAvg on average
   (Wilcoxon, Holm-adjusted p = 0.003), and was the only strategy to reach T95 in every seed of
   P0, P1 and P4 (60–76 rounds). Its gain was as large under IID data (P0: 0.960 vs 0.883) as
   under skew, so it acted as a faster server optimizer rather than a fix for heterogeneity.
3. **FedProx (μ = 0.01) gave the same results as FedAvg** (differences ≤ 0.004). **FedNova gave
   the same results as FedAvg under P0**, where all clients take the same number of local steps,
   and was slightly but consistently **worse** under skew (mean −0.021; 9 of 12 blocks worse;
   Holm p = 0.032).
4. **Attack-type exclusivity (P3) was the hardest partition for every strategy.** On the random
   protocol the best P3 score was 0.773 (FedAdam), against 0.971 centralized. FedAvg, FedProx and
   FedAdam had BFA recall ≈ 0 under P3.
5. **Without attackers, the robust rules lost substantially under non-IID partitions.** Krum fell
   to 0.222 under P3, Median to 0.482 and Trimmed Mean to 0.541 (random protocol), and all three
   were significantly worse than FedAvg overall. Under P0 they matched FedAvg (0.874–0.888).
6. **Poisoning (chronological, P4, 2 of 8 malicious) hurt FedAvg little.** Sign flip lowered
   FedAvg macro-F1 from 0.468 to 0.426, and targeted relabeling to 0.419, with attack FNR rising
   only from 0.033 to 0.043. The robust rules did not protect better: under targeted relabeling
   their attack FNR was higher than FedAvg's (Median 0.200, Trimmed Mean 0.133, Krum 0.394).
   Variance across seeds was large.

---

## 2. Environment and reproducibility

### Hardware and OS

- One laptop CPU: **Intel Core i7-7600U @ 2.80 GHz, 2 cores / 4 threads**. WSL reports 7 GB RAM;
  the Windows host's total RAM is not recorded. No GPU was used (CPU build of PyTorch).
- **Windows** host with a native venv (`.venv`, Python 3.13.17) used for unit tests, and **WSL 2**
  (Ubuntu 26.04.1 LTS, kernel 6.18.54.1-microsoft-standard-WSL2) with venv `~/venvs/fedsdn`
  (Python 3.13.18, same version as Kaggle) used for simulation. The logs do not record which of
  the two environments launched the experiment runs; the WSL venv has the complete lock file
  (`requirements-lock-wsl.txt`) and is the one used to re-run the tests for this write-up.

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
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| pytest | 9.2.0 |
| flwr[simulation] | 1.43.0 (installed; not used by the reduced study) |
| ray | 2.55.1 (WSL only; not used by the reduced study) |

Full freezes: `requirements-lock.txt` (Windows), `requirements-lock-wsl.txt` (WSL).

### Simulation design

- `src/fedsdn/sim.py`: plain PyTorch, **no Flower and no Ray**. In each round the K = 8 clients
  are trained **one after another in the same process**, starting from the same global weights;
  the server rule then combines the K flattened updates.
- `scripts/run_reduced.py` runs independent runs in parallel with a pool of **4 worker
  processes** (`multiprocessing`, spawn context, `torch.set_num_threads(1)` per worker).
- Resumability: each finished run writes `<run_id>_rounds.csv` and then `<run_id>.json`
  (atomic writes). The JSON is the "done" marker, and finished runs are skipped on restart.
  Progress is logged to `results/progress.log`.
- Paths come from `FEDSDN_DATA` (default `./data`) and `FEDSDN_RESULTS` (default `./results`).

### Commit for each stage

| stage | commit | date (+05:00) |
|---|---|---|
| Project skeleton, pins, original rules (incl. 60 s gap split) | `d3d276c` | 2026-11-05 22:51 |
| WSL environment, Flower smoke test | `acfad95` | 2026-11-06 07:56 |
| InSDN preprocessing, per-family minute-cut split | `2f6d43d` | 2026-11-06 08:25 |
| Partitioning P0–P4, federated scaling | `5fa6680` | 2026-11-06 08:38 |
| Simulation, strategies, both protocols, report (R = 30 runs) | `1792d65` | 2026-11-06 09:33 |
| R = 100 random protocol, FedAvg + FedAdam | `d83efb2` | 2026-11-06 10:25 |
| R = 100 random protocol, all strategies | `2c2a127` | 2026-11-06 10:57 |
| Paper statistics, figures, tables | `d423380` | 2026-11-06 11:04 |
| Wilcoxon vs FedAvg | `7bdeb48` | 2026-11-06 11:18 |
| Architecture / model / pipeline diagrams | `51eb753` | 2026-11-06 11:22 |

The runs were started from the working tree shortly before each commit (e.g. the R = 30 runs ran
09:03–09:32, and their code was committed at 09:33). The simulation code in `1792d65`–`51eb753` is
the code that produced the results. The later commit `6c241e8` (Flower validation, excluded from
this document) adds an option to `sim.py` that is off by default and does not affect these
results.

Data fingerprints: `flows.parquet` SHA-256 `fbf3ad7570b8…a52c` (chronological). Raw CSV SHA-256s
are stored in `data/processed/insdn/meta.json`. Every partition sidecar stores the
`flows.parquet` hash, and the partition scripts refuse to run on a mismatch.

### Commands to reproduce everything

```bash
# WSL venv (see CLAUDE.md for creating it), from the repo root
source ~/venvs/fedsdn/bin/activate
export FEDSDN_DATA=./data FEDSDN_RESULTS=./results
# raw CSVs in data/raw/insdn/{Normal_data,OVS,metasploitable-2}.csv

python -m pytest                                              # 49 unit tests
python scripts/build_dataset.py --dataset insdn               # processed/insdn
python scripts/make_partitions.py --dataset insdn --K 8 --seeds 0-4      # all 10 settings x 5 seeds
python scripts/make_random_split.py --dataset insdn --seed 0  # processed/insdn_random
python scripts/make_partitions.py --dataset insdn --variant random --seeds 0-2 \
    --settings P0 P1_a0.5 P3_m1 P4_a0.5b0.5                   # partitions/insdn_random
python scripts/run_reduced.py --dry          # optional: 18 runs at R = 2 -> results/dry/
python scripts/run_reduced.py                # 198 runs: chrono R = 30 (111) + random R = 30 (87)
python scripts/run_reduced.py --suite r100   # 87 runs: random R = 100 -> results/r100/random/
python scripts/report_reduced.py             # results/report/tables.md + figures
python scripts/paper_outputs.py              # results/paper_tables/*.csv, results/paper_figures/fig1-6
python scripts/paper_diagrams.py             # fig0_system_architecture, fig_model_architecture, fig_pipeline
python scripts/diag_central.py               # centralized diagnostics (console only)
```

Runs are deterministic given the code and data. The federated final-round values of an earlier,
superseded chronological batch (Section 8) are identical to those of the final batch.

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
  rows by q_c. α ∈ {0.1, 0.5, 1, 10} built; **α = 0.5 used**.
- **P2 (quantity skew):** one s ~ Dir_K(β); all rows dealt by s, so the class mix of each shard
  matches the global mix in expectation. β ∈ {0.5, 1} built; **not used**.
- **P3 (attack-type exclusivity):** benign rows dealt equally over all K clients; the attack
  families are put in a seeded random order and assigned round-robin, with
  m_eff = max(m, ⌈F/K⌉) families per client (F = 5 attack families, K = 8 → m_eff = 1 for m = 1).
  Families are assigned with replacement, so every family has at least one holder; a family
  is dealt equally among its holders. m ∈ {1, 2} built; **m = 1 used**.
- **P4 (combined skew, product reweighting):** an approved addition not specified in the paper
  beyond "α = β = 0.5". One s ~ Dir_K(0.5) for shard sizes, and per family q_c ~ Dir_K(0.5).
  Family c is dealt by q̃_{c,k} = q_{c,k}·s_k / Σ_j q_{c,j}·s_j. With uniform s this reduces to
  P1; with uniform q it reduces to P2 (in expectation), as checked by a unit test. **Used.**

Built: all 10 settings × seeds 0–4 for the chronological protocol (50 partitions). For the random
protocol only the 4 used settings × seeds 0–2 were built. **Used in runs: P0, P1 α = 0.5,
P3 m = 1, P4 α = β = 0.5, seeds 0–2, K = 8.**

### Minimum shard size

Every shard (validation rows included) must have n_min = 100 rows. Up to 100 fresh draws are
tried. If all fail, the draw with the largest minimum shard is kept and random rows are moved from
the current largest shard. In practice every used partition met the minimum within 1–3 draws
(P4 needed 2–3 draws for some seeds), and **no rows were moved** in any used partition.

### Validation hold-out

Per client, round(0.1·n_k) rows are held out for validation. They are spread over families in
proportion to the client's family counts (largest remainder) and drawn at random within each
family. Validation rows are excluded from training, from the standardization statistics and from
the class-weight counts.

### Realized heterogeneity (used settings, K = 8)

JSD = mean pairwise Jensen–Shannon divergence (base 2) of the clients' family distributions,
computed over whole shards. CV = coefficient of variation of shard sizes (population std).

| protocol | partition | seed | JSD | CV | min shard | max shard | draws |
|---|---|---|---|---|---|---|---|
| chrono | P0 | 0 / 1 / 2 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 6,533 | 6,534 | 1 |
| chrono | P1 α0.5 | 0 / 1 / 2 | 0.319 / 0.343 / 0.351 | 0.359 / 0.583 / 0.558 | 2,643 / 906 / 890 | 9,764 / 13,584 / 10,440 | 1 |
| chrono | P3 m1 | 0 / 1 / 2 | 0.485 / 0.531 / 0.485 | 0.868 / 0.610 / 0.868 | 2,160 | 18,164 / 10,917 / 18,164 | 1 |
| chrono | P4 | 0 / 1 / 2 | 0.419 / 0.343 / 0.345 | 0.924 / 1.392 / 0.915 | 174 / 213 / 256 | 14,301 / 28,230 / 19,801 | 2 / 1 / 2 |
| random | P0 | 0 / 1 / 2 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 6,326 | 6,327 | 1 |
| random | P1 α0.5 | 0 / 1 / 2 | 0.371 / 0.354 / 0.426 | 0.488 / 0.263 / 0.669 | 1,337 / 3,582 / 1,288 | 10,682 / 9,063 / 15,273 | 1 |
| random | P3 m1 | 0 / 1 / 2 | 0.482 / 0.527 / 0.484 | 0.858 / 0.582 / 0.859 | 2,291 / 2,291 / 2,145 | 18,000 / 10,000 / 18,000 | 1 |
| random | P4 | 0 / 1 / 2 | 0.359 / 0.222 / 0.402 | 0.819 / 1.144 / 1.288 | 129 / 235 / 113 | 16,364 / 21,427 / 24,961 | 2 / 2 / 3 |

P3 had the highest label skew (JSD 0.48–0.53). P4 had the highest quantity skew (CV up to 1.39)
with label skew between P1 and P3.

### P3 per-client family layout (chronological protocol; shard rows incl. validation)

With 5 attack families over 8 clients and m_eff = 1, three families have two holders and two
families have one. Because Rare is one of the 5 "attack" families, **one or two clients per seed
hold only Normal + Rare**, which means they train on **Normal only** in the multi-class task
(Rare is masked).

| seed | family → clients |
|---|---|
| 0 | BFA → {2, 7}; DDoS → {3}; DoS → {1, 6}; Probe → {4}; Rare → {0, 5} |
| 1 | BFA → {3}; DDoS → {4}; DoS → {1, 6}; Probe → {2, 7}; Rare → {0, 5} |
| 2 | BFA → {2, 7}; DDoS → {4}; DoS → {0, 5}; Probe → {3}; Rare → {1, 6} |

Normal is spread equally (1,998–1,999 rows per client). Example, seed 0 (Normal / attack rows):
c0 1,999 / Rare 162; c1 1,999 / DoS 8,918; c2 1,999 / BFA 544; c3 1,998 / DDoS 871;
c4 1,998 / Probe 16,166; c5 1,998 / Rare 162; c6 1,998 / DoS 8,918; c7 1,998 / BFA 543.

Random protocol P3: seed 0: BFA {2, 7}, DDoS {0, 5}, DoS {3}, Probe {1, 6}, Rare {4}; seed 1:
BFA {3}, DDoS {2, 7}, DoS {0, 5}, Probe {1, 6}, Rare {4}; seed 2: BFA {2, 7}, DDoS {4},
DoS {0, 5}, Probe {3}, Rare {1, 6}. Normal: 2,000 rows per client.

---

## 5. Model and training

- **Model** (`src/fedsdn/models.py`): MLP 68 → 128 → 64 → 5, ReLU, dropout 0.2 after each hidden
  layer, logits output with `CrossEntropyLoss`. **17,413 parameters** (8,832 + 8,256 + 325),
  68 KB in float32. The secondary 1D-CNN was not run.
- **Local optimizer:** plain mini-batch SGD, learning rate 0.05, **momentum 0, no weight decay**,
  batch size 256, E = 1 local epoch per round. Local steps per round
  τ_k = E·⌈n_k / 256⌉, where n_k is the client's training rows after masking Rare. A fresh
  optimizer is created every round, so there is no state across rounds.
- **Class weights (global balanced):** each client reports its training-row class counts
  (validation excluded, Rare masked); the server sums them and sets
  w_c = n_train / (C·n_c), with C = 5. They are computed once per run, before round 1, and
  the same weights are sent to every client. Approximate values on the chronological training pool
  **[computed for this write-up from pool counts; the runs used the counts without the 10%
  validation rows]**: Normal 0.65, BFA 9.56, DDoS 11.93, DoS 0.58, Probe 0.64.
- **Federated standardization** (`src/fedsdn/data/scaling.py`): each client sends count, per-feature
  sum and sum of squares over its training rows (validation excluded); the server computes the
  global mean and population std in float64. A feature with variance ≤ 1e-12·E[x²] is treated as
  constant and gets std 1. The same mean / std are applied to every row (train, validation, test).
  A unit test checks that this equals pooled NumPy standardization.
- **Rare rows (y = −1)** are dropped from each client's training rows and from validation and test
  evaluation, which is equivalent to masking them. They are included in the standardization
  statistics.
- **Rounds:** R = 30 (chronological, and poisoning), R = 100 (random protocol headline). Earlier
  random-protocol runs at R = 30 are kept only for comparison.
- **Centralized baseline:** the same model, optimizer, batch size, class weighting and
  standardization (one "client" with the whole pool). It holds out a stratified 10% of the training
  pool as validation, and trains for as many epochs as the protocol has rounds (30 or 100). It
  reports the test score at the best-validation epoch.
- **Seeds 0, 1, 2.** The seed selects the partition file (one partition per seed) and sets
  `torch.manual_seed` (model initialization and dropout). Batch order uses a separate generator per
  (seed, round, client): `seed·1,000,003 + round·101 + client`. Malicious clients are drawn with
  `default_rng([seed, 2024])`.
- **No hyperparameter tuning** was done (Section 13).

---

## 6. Aggregation strategies

Implemented in `src/fedsdn/strategies/aggregate.py` (server rules) and `src/fedsdn/sim.py` (client
side). Each client returns its update Δ_k = w_k − w (flattened). The server computes an aggregate
Δ and sets w ← w + Δ. p_k = n_k / Σ_j n_j, with n_k = the client's training rows (validation
excluded, Rare masked).

| strategy | rule (in words) | hyperparameters | weighting |
|---|---|---|---|
| **FedAvg** | Δ = Σ p_k Δ_k | — | by sample count |
| **FedProx** | Same server rule as FedAvg. Each client adds (μ/2)‖w − w_global‖² to its local loss. | μ = 0.01 | by sample count |
| **FedNova** | Normalize each update by its local step count, d_k = Δ_k / τ_k, then rescale: Δ = τ_eff·Σ p_k d_k, with τ_eff = Σ p_k τ_k. Equals FedAvg when all τ_k are equal. | — (plain SGD so τ_k is exact) | by sample count |
| **FedAdam** | Pseudo-gradient g = FedAvg Δ; m ← β1·m + (1−β1)·g; v ← β2·v + (1−β2)·g²; Δ = η_s·m / (√v + ε). m, v start at 0; no bias correction (as in the paper's formula). | η_s = 0.01, β1 = 0.9, β2 = 0.99, ε = 1e-3 | by sample count |
| **Krum** | For each update, sum the squared distances to its K − f̂ − 2 = 4 nearest other updates; the update with the smallest sum becomes the global update. | f̂ = 2 | equal (one update selected) |
| **Median** | Coordinate-wise median of the 8 updates (with even K, the mean of the two middle values). | — | equal |
| **Trimmed Mean** | Per coordinate, drop the f̂ = 2 largest and 2 smallest values and average the remaining 4. | f̂ = 2 | equal |

- f̂ = ⌊K/4⌋ = 2 was used in **every** condition, honest ones included, as the paper specifies (a
  defender does not know whether an attack is under way). K = 8 satisfies Krum (8 ≥ 2f̂ + 3 = 7)
  and Trimmed Mean (8 > 2f̂ = 4).
- The robust rules weight clients equally. FedAvg, FedProx, FedNova and FedAdam weight them by
  sample count.
- SCAFFOLD was not implemented.
- **Poisoning** (in `sim.py`). Malicious clients are fixed per seed: {0, 3} for seeds 0 and 2,
  {1, 4} for seed 1.
  - *sign flip*: the client sends −Δ_k.
  - *targeted*: before training, every attack-labelled training row on the malicious client is
    relabelled Normal.
  - Malicious clients' validation sets keep their true labels.
  - Random label flipping was not run.

### Tests (`tests/`, 49 tests, all passing; re-run for this write-up: `49 passed`)

- `test_aggregate.py`:
  - FedNova equals FedAvg when all τ_k are equal (random 8 × 50 updates, unequal n_k);
  - FedNova differs with unequal steps (hand-computed 4/3);
  - Median and Trimmed Mean on known vectors (even and odd K, f = 1, 2);
  - Krum rejects an outlier.
- `test_partition.py`: exact largest-remainder sizes; Rare mapping in the training pool;
  every row assigned exactly once and deterministically; P4 reduces to P1 / P2; P4 combines both
  skews; P0 balance; smaller α → higher JSD; P2 keeps the label mix; P3 holders cover every family
  and are exclusive / evenly split; the minimum-shard retry and row-move logic; the 10% stratified
  validation; partition file writing; **federated scaling equals pooled NumPy** and excludes
  validation rows.
- `test_preprocess.py`: conflict groups dropped entirely and earliest duplicate kept; the
  rare rule (below, at and without threshold); the time-ordered cap; minute-cut rules (closest to
  target, never splits a minute, inclusive bounds, minimum test size); grouped fallback with
  unordered pairs; relaxed cut up to 50% (a DDoS-like case) and its precedence over the grouped
  split; pairs-per-flow and the degenerate flag; rare families and the all-train fallback; both
  timestamp formats; an end-to-end check.
- The paper's planned SCAFFOLD test (equals FedAvg with zero control variates) does not exist,
  because SCAFFOLD was not implemented.

---

## 7. Evaluation

### Protocols

- **Chronological (primary):** the leakage-aware per-family minute-cut split of Section 3
  (`processed/insdn`, `partitions/insdn`). R = 30.
- **Random (secondary):** a random stratified 80/20 split of the same non-capped rows, stratified by
  partition family, seed 0 (`processed/insdn_random`, `partitions/insdn_random`). R = 100.

**Why the random protocol was added.** The centralized baseline on the chronological split reached
only 0.521 macro-F1. `scripts/diag_central.py` (seed 0, console output only; numbers as recorded in
`CLAUDE.md`) showed:

- the model was **not underfitting** (training macro-F1 0.88);
- it was **not undertrained** (100 epochs at lr 0.1 still gave test 0.52);
- a random split of the same rows gave **0.95**.

The gap is therefore distribution shift between the time periods / address pairs in train and test:
BFA test recall is 0 (held-out address pair) and Probe ≈ 0.3. With most of the error coming from
that shift, the chronological test set separates strategies poorly. The random protocol was added
as a second view in which strategy differences are visible, but it has the flow-level leakage the
paper warns about. **Its absolute scores are inflated and must not be compared with chronological
scores.** It is used only to rank strategies against each other.

### Metrics (computed every round on the global test set; multi-class, Rare masked)

- **Macro-F1** over the 5 classes (primary). Also accuracy and **per-class recall**.
- **FPR** = share of Normal test flows predicted as any attack class.
- **Attack FNR** = share of attack test flows predicted as Normal (used for poisoning).
- **Rare-class recall** = recall of the **single** class with the smallest global training count
  in the run: **DDoS** on the chronological protocol and **BFA** on the random protocol.
- **Validation macro-F1** (global): computed on the pooled validation sets of all clients (pooled
  predictions and labels, i.e. sample-weighted).
- **Worst-client validation macro-F1:** per client, macro-F1 over the classes present in its own
  validation set; minimum over clients.
- **Validation-selected round (headline):** the round with the best global validation macro-F1
  (ties → earliest). The run reports test metrics at that round. The test set is never used for
  selection. Also reported: the **final round**, and **AUC** = mean test macro-F1 over rounds
  1..R.
- **T95:** the first round whose test macro-F1 is ≥ 0.95 × the protocol's mean centralized
  validation-selected test macro-F1, with the centralized model trained for as many epochs as the
  protocol has rounds. Thresholds:
  - chronological R = 30: 0.95 × 0.5207 = **0.4947**;
  - random R = 100: 0.95 × 0.9706 = **0.9221**.

  Runs that never reach it are censored. Tables report the mean T95 over the seeds that reach it,
  plus the number of censored seeds.
- **Bytes per round** (derived, not measured): params × 4 B × 2K = 17,413 × 4 × 16 =
  **1,114,432 B (1.11 MB)**, the same for all 7 strategies. **MB to T95** = T95 × 1.114 MB.
- **Time:** each round's wall-clock time is logged per run (`wall_s`). Client CPU time and server
  aggregation time were **not** measured separately. With 4 worker processes sharing 2 physical
  cores, wall time is not a clean cost measure.

---

## 8. Run inventory

| batch | what | runs | elapsed (4 workers) | where | status |
|---|---|---|---|---|---|
| Dry run (v1) | early smoke test before validation logging | not recorded | not recorded | `results/archive_chrono_v1_noval/dry/` | superseded |
| Chrono v1 | chrono R = 30: 3 centralized + 84 federated + 24 poisoning; **no validation logging**, headline was final-round F1 | 111 | 14.0 min (finished 08:55) | `results/archive_chrono_v1_noval/` | **superseded**: no validation logging, so no validation-based round selection |
| Dry run | R = 2: per protocol, centralized + 7 strategies on P4 seed 0; 2 poisoning runs (Median, chrono) | 18 | 20 s (09:03:22–09:03:42) | `results/dry/` | test only; never used for outputs |
| Main | chrono R = 30 (3 centralized 30 epochs + 84 federated + 24 poisoning) **and** random R = 30 (3 centralized + 84 federated) | 198 (111 + 87) | 29.0 min (09:03:52–09:32:56) | `results/chrono/`, `results/random/` | chrono: **final**. random R = 30: **superseded** as headline, kept for the R = 30 vs R = 100 comparison and the leakage diagnostic |
| R = 100, part 1 | random, FedAvg + FedAdam × 4 partitions × 3 seeds | 24 | 8.6 min (10:03:17–10:11:52) | `results/r100/random/` | final |
| R = 100, part 2 | random, centralized 100 epochs (3) + FedProx, FedNova, Krum, Median, Trimmed Mean × 12 | 63 | 21.5 min (10:35:58–10:57:25) | `results/r100/random/` | final (87 runs with part 1) |
| Centralized diagnostics | `diag_central.py`, seed 0: chrono 30 ep lr 0.05; random 30 ep; chrono 100 ep lr 0.1 | 3 | not recorded | console only | diagnostic; numbers in Section 7 |

- **Why R = 30 was superseded for the random protocol:** at R = 30 the validation-selected rounds
  sat near the end of training, and in the R = 30 report (`results/report/t95_random.csv`) T95 was
  censored in 83 of 84 federated runs. (The threshold used for that file is not stored in it.)
  The runs had not converged.
- Per-run wall time (mean): chrono federated 25.7 s (R = 30), chrono poisoning 32–34 s, random
  R = 30 federated 43.5 s, random R = 100 federated 80.5 s, centralized 100 epochs 99.3 s. Sum of
  per-run times: chrono 0.84 h, random R = 30 1.07 h, random R = 100 1.96 h.
- Determinism check: the federated final-round values in the v1 archive are identical to the
  final-round values of the main batch (e.g. FedAvg P0 0.410, FedNova P3 0.571, Krum P3 0.154).
- **Final results used in the paper:** `results/chrono/` (111 runs, R = 30) and
  `results/r100/random/` (87 runs, R = 100). `results/random/` (R = 30) is used only for the
  R = 30 vs R = 100 table and for the leakage diagnostic (30 epochs on both protocols).
- The Flower validation run (`results/flower_validation*`) is excluded here (see the last
  section).

---

## 9. Results

All values are mean ± std over seeds 0–2, test macro-F1 at the validation-selected round unless
stated. "chrono" = chronological, R = 30. "random" = random split, R = 100. **Do not compare
chrono and random numbers with each other.**

### 9.1 Leakage diagnostic (centralized, 30 epochs on both protocols)

| protocol | macro-F1 | Normal | BFA | DDoS | DoS | Probe |
|---|---|---|---|---|---|---|
| chronological | 0.521 ± 0.001 | 0.994 | 0.000 | 1.000 | 0.797 | 0.301 |
| random | 0.924 ± 0.023 | 0.986 | 0.802 | 0.997 | 0.986 | 0.942 |

The same model, data rows and budget lose 0.40 macro-F1 when the test set comes from later
minutes / an unseen address pair. The loss is concentrated in BFA (0 recall) and Probe (0.30).
DDoS and Normal are easy on both. With 100 epochs, the random centralized baseline reached
0.971 ± 0.001.

### 9.2 Headline: validation-selected test macro-F1

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.410 ± 0.047 | 0.516 ± 0.020 | 0.634 ± 0.034 | 0.468 ± 0.059 | 0.883 ± 0.010 | 0.892 ± 0.011 | 0.763 ± 0.008 | 0.908 ± 0.047 |
| FedProx | 0.409 ± 0.044 | 0.520 ± 0.013 | 0.634 ± 0.034 | 0.466 ± 0.057 | 0.882 ± 0.010 | 0.892 ± 0.010 | 0.763 ± 0.008 | 0.906 ± 0.044 |
| FedNova | 0.411 ± 0.048 | 0.516 ± 0.011 | 0.590 ± 0.054 | 0.465 ± 0.074 | 0.883 ± 0.010 | 0.875 ± 0.007 | 0.746 ± 0.034 | 0.860 ± 0.044 |
| FedAdam | 0.460 ± 0.079 | 0.418 ± 0.087 | 0.623 ± 0.032 | 0.465 ± 0.071 | **0.960 ± 0.006** | **0.952 ± 0.010** | **0.773 ± 0.005** | **0.964 ± 0.008** |
| Krum | **0.540 ± 0.036** | 0.456 ± 0.159 | 0.168 ± 0.053 | 0.250 ± 0.062 | 0.888 ± 0.005 | 0.663 ± 0.248 | 0.222 ± 0.061 | 0.618 ± 0.114 |
| Median | 0.417 ± 0.055 | **0.525 ± 0.148** | 0.264 ± 0.099 | 0.564 ± 0.045 | 0.874 ± 0.015 | 0.813 ± 0.014 | 0.482 ± 0.167 | 0.763 ± 0.061 |
| Trimmed Mean | 0.429 ± 0.077 | 0.507 ± 0.122 | 0.295 ± 0.065 | **0.565 ± 0.012** | 0.875 ± 0.015 | 0.822 ± 0.011 | 0.541 ± 0.118 | 0.784 ± 0.059 |
| Centralized | 0.521 ± 0.001 | | | | 0.971 ± 0.001 | | | |

- **Random protocol:** FedAdam was best in every partition. FedAvg, FedProx and FedNova were close
  to each other. The robust rules matched FedAvg under P0 but fell well behind under P1, P3 and
  P4, with Krum worst. P3 was the hardest partition for every strategy (best 0.773 vs 0.971
  centralized).
- **Chronological protocol:** all scores lie between 0.17 and 0.64, against 0.52 centralized. The
  ordering changes from partition to partition (Krum best under P0, FedAvg / FedProx under P3,
  Median / Trimmed Mean under P4), and several federated settings beat the centralized model.
  This is consistent with a test set dominated by shift (Section 9.1), where which classes happen
  to transfer matters more than the aggregation rule.

Final-round and AUC values (`table_final_auc_macro_f1.csv`) follow the same pattern. On the random
protocol, final-round scores are 0.01–0.08 below the validation-selected ones (e.g. FedAvg P0:
final 0.849, AUC 0.791; FedAdam P0: final 0.912, AUC 0.835; centralized: final 0.926,
AUC 0.890).

### 9.3 Per-class test recall at the validation-selected round

DoS / Probe / DDoS:

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.20/0.32/1.00 | 0.72/0.33/1.00 | 0.81/0.50/0.99 | 0.47/0.31/1.00 | 0.96/0.92/0.99 | 0.98/0.96/0.99 | 0.99/0.92/0.91 | 0.97/0.95/0.98 |
| FedProx | 0.19/0.32/1.00 | 0.74/0.32/1.00 | 0.81/0.50/0.99 | 0.46/0.31/1.00 | 0.96/0.92/0.99 | 0.98/0.96/0.99 | 0.99/0.92/0.91 | 0.97/0.95/0.98 |
| FedNova | 0.20/0.32/1.00 | 0.71/0.32/1.00 | 0.79/0.45/1.00 | 0.48/0.31/1.00 | 0.96/0.92/0.99 | 0.97/0.92/0.99 | 0.77/0.92/0.96 | 0.92/0.91/0.99 |
| FedAdam | 0.48/0.30/1.00 | 0.31/0.30/1.00 | 0.80/0.62/0.99 | 0.50/0.30/1.00 | 0.99/0.99/1.00 | 0.98/0.99/1.00 | 0.99/0.93/0.96 | 0.99/0.99/1.00 |
| Krum | 0.75/0.33/1.00 | 0.62/0.47/1.00 | 0.00/0.00/0.00 | 0.37/0.29/0.67 | 0.98/0.92/0.99 | 0.61/0.58/0.96 | 0.00/0.00/0.32 | 0.74/0.63/0.65 |
| Median | 0.23/0.32/1.00 | 0.55/0.45/1.00 | 0.52/0.10/0.00 | 0.38/0.61/1.00 | 0.96/0.92/0.99 | 0.84/0.88/0.97 | 0.36/0.41/0.66 | 0.68/0.82/0.96 |
| Trimmed Mean | 0.30/0.32/1.00 | 0.55/0.45/1.00 | 0.80/0.10/0.00 | 0.58/0.52/1.00 | 0.96/0.92/0.99 | 0.86/0.90/0.98 | 0.37/0.72/0.67 | 0.74/0.84/0.97 |
| Centralized | 0.80/0.30/1.00 | | | | 0.99/0.99/1.00 | | | |

BFA:

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.00 | 0.00 | 0.00 | 0.00 | 0.77 | 0.57 | 0.00 | 0.71 |
| FedProx | 0.00 | 0.00 | 0.00 | 0.00 | 0.77 | 0.57 | 0.00 | 0.71 |
| FedNova | 0.00 | 0.00 | 0.00 | 0.00 | 0.77 | 0.58 | 0.26 | 0.67 |
| FedAdam | 0.00 | 0.00 | 0.00 | 0.00 | 0.83 | 0.77 | 0.01 | 0.80 |
| Krum | 0.00 | 0.00 | 0.67 | 0.00 | 0.56 | 0.82 | 0.67 | 0.61 |
| Median | 0.00 | 0.00 | 0.00 | 0.00 | 0.65 | 0.70 | 0.49 | 0.70 |
| Trimmed Mean | 0.00 | 0.00 | 0.00 | 0.00 | 0.65 | 0.71 | 0.26 | 0.68 |
| Centralized | 0.00 | | | | 0.85 | | | |

Normal recall was 0.977–1.000 everywhere except Krum (chrono P1 0.56, chrono P4 0.31; random
0.966–0.988).

- On the random protocol, **BFA, the class with the fewest training rows, is the first to
  degrade**: FedAvg BFA recall falls from 0.77 (P0) to 0.57 (P1) and 0.00 (P3), while DoS,
  Probe and DDoS stay ≥ 0.91.
- Under P3, Krum collapses onto few classes (DoS and Probe recall 0), and Median / Trimmed Mean
  lose DDoS and DoS.
- On the chronological protocol, BFA is 0 for nearly every strategy and Probe stays near 0.30–0.6,
  which is the shift seen in 9.1. (Krum's 0.67 BFA recall under chrono P3 comes with 0 recall on
  DDoS, DoS and Probe.)

### 9.4 T95 and bytes

Cells: mean T95 rounds over seeds that reached it / MB to T95; "cens. n/3" = seeds that never
reached it. 1.114 MB per round for every strategy.

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 3.5 r / 3.9 MB (cens. 1/3) | 2.3 r / 2.6 MB | 24.7 r / 27.5 MB | 3.0 r / 3.3 MB | cens. 3/3 | cens. 3/3 | cens. 3/3 | 82.0 r / 91.4 MB (cens. 1/3) |
| FedProx | 3.5 r / 3.9 MB (cens. 1/3) | 2.3 r / 2.6 MB | 24.7 r / 27.5 MB | 3.0 r / 3.3 MB | cens. 3/3 | cens. 3/3 | cens. 3/3 | 82.0 r / 91.4 MB (cens. 1/3) |
| FedNova | 3.5 r / 3.9 MB (cens. 1/3) | 2.0 r / 2.2 MB | 6.7 r / 7.4 MB | 1.3 r / 1.5 MB | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 |
| FedAdam | 9.7 r / 10.8 MB | 2.7 r / 3.0 MB | 16.3 r / 18.2 MB | 3.0 r / 3.3 MB | **76.0 r / 84.7 MB** | **65.3 r / 72.8 MB** | cens. 3/3 | **60.3 r / 67.2 MB** |
| Krum | 10.0 r / 11.1 MB | 18.0 r / 20.1 MB (cens. 2/3) | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 |
| Median | 4.7 r / 5.2 MB | 3.5 r / 3.9 MB (cens. 1/3) | cens. 3/3 | 6.7 r / 7.4 MB | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 |
| Trimmed Mean | 4.7 r / 5.2 MB | 3.0 r / 3.3 MB | cens. 3/3 | 7.0 r / 7.8 MB | cens. 3/3 | cens. 3/3 | cens. 3/3 | cens. 3/3 |

- On the random protocol, **only FedAdam** reached 95% of centralized in all seeds of P0, P1 and
  P4 (60–76 rounds, 67–85 MB). FedAvg / FedProx reached it in 2 of 3 P4 seeds (82 rounds). No
  strategy reached it under P3.
- On the chronological protocol the threshold is low (0.495), so most settings cross it within a
  few rounds. These T95 values say little about convergence.
- Because all strategies send messages of the same size, bytes only restate rounds.

### 9.5 Random protocol: R = 30 vs R = 100

Both T95 columns use the R = 100 threshold (0.9221). "Sel. round" = validation-selected round at
R = 100, mean (min–max).

| strategy / partition | R = 30 | R = 100 | sel. round (R = 100) | T95 R = 30 | T95 R = 100 |
|---|---|---|---|---|---|
| FedAvg P0 | 0.788 ± 0.007 | 0.883 ± 0.010 | 90 (74–99) | cens. 3/3 | cens. 3/3 |
| FedAvg P1 | 0.792 ± 0.028 | 0.892 ± 0.011 | 93 (92–93) | cens. 3/3 | cens. 3/3 |
| FedAvg P3 | 0.731 ± 0.008 | 0.763 ± 0.008 | 90 (83–95) | cens. 3/3 | cens. 3/3 |
| FedAvg P4 | 0.810 ± 0.016 | 0.908 ± 0.047 | 87 (74–99) | cens. 3/3 | 82.0 r (cens. 1/3) |
| FedProx P0 | 0.788 ± 0.007 | 0.882 ± 0.010 | 90 (74–99) | cens. 3/3 | cens. 3/3 |
| FedProx P1 | 0.793 ± 0.029 | 0.892 ± 0.010 | 93 (92–93) | cens. 3/3 | cens. 3/3 |
| FedProx P3 | 0.731 ± 0.008 | 0.763 ± 0.008 | 92 (86–97) | cens. 3/3 | cens. 3/3 |
| FedProx P4 | 0.811 ± 0.017 | 0.906 ± 0.044 | 87 (74–99) | cens. 3/3 | 82.0 r (cens. 1/3) |
| FedNova P0 | 0.788 ± 0.007 | 0.883 ± 0.010 | 90 (74–99) | cens. 3/3 | cens. 3/3 |
| FedNova P1 | 0.801 ± 0.008 | 0.875 ± 0.007 | 88 (80–93) | cens. 3/3 | cens. 3/3 |
| FedNova P3 | 0.700 ± 0.062 | 0.746 ± 0.034 | 79 (75–83) | cens. 3/3 | cens. 3/3 |
| FedNova P4 | 0.799 ± 0.034 | 0.860 ± 0.044 | 92 (81–98) | cens. 3/3 | cens. 3/3 |
| FedAdam P0 | 0.834 ± 0.018 | 0.960 ± 0.006 | 89 (84–96) | cens. 3/3 | 76.0 r |
| FedAdam P1 | 0.832 ± 0.019 | 0.952 ± 0.010 | 94 (92–97) | cens. 3/3 | 65.3 r |
| FedAdam P3 | 0.753 ± 0.019 | 0.773 ± 0.005 | 72 (49–86) | cens. 3/3 | cens. 3/3 |
| FedAdam P4 | 0.861 ± 0.034 | 0.964 ± 0.008 | 95 (92–99) | cens. 3/3 | 60.3 r |
| Krum P0 | 0.797 ± 0.013 | 0.888 ± 0.005 | 88 (83–93) | cens. 3/3 | cens. 3/3 |
| Krum P1 | 0.459 ± 0.345 | 0.663 ± 0.248 | 71 (53–87) | cens. 3/3 | cens. 3/3 |
| Krum P3 | 0.157 ± 0.052 | 0.222 ± 0.061 | 39 (16–83) | cens. 3/3 | cens. 3/3 |
| Krum P4 | 0.410 ± 0.063 | 0.618 ± 0.114 | 89 (69–99) | cens. 3/3 | cens. 3/3 |
| Median P0 | 0.789 ± 0.008 | 0.874 ± 0.015 | 96 (90–99) | cens. 3/3 | cens. 3/3 |
| Median P1 | 0.757 ± 0.036 | 0.813 ± 0.014 | 84 (80–91) | cens. 3/3 | cens. 3/3 |
| Median P3 | 0.404 ± 0.127 | 0.482 ± 0.167 | 83 (73–99) | cens. 3/3 | cens. 3/3 |
| Median P4 | 0.591 ± 0.034 | 0.763 ± 0.061 | 99 (97–100) | cens. 3/3 | cens. 3/3 |
| Trimmed Mean P0 | 0.789 ± 0.008 | 0.875 ± 0.015 | 96 (90–99) | cens. 3/3 | cens. 3/3 |
| Trimmed Mean P1 | 0.762 ± 0.022 | 0.822 ± 0.011 | 95 (91–100) | cens. 3/3 | cens. 3/3 |
| Trimmed Mean P3 | 0.455 ± 0.146 | 0.541 ± 0.118 | 70 (38–94) | cens. 3/3 | cens. 3/3 |
| Trimmed Mean P4 | 0.691 ± 0.054 | 0.784 ± 0.059 | 86 (70–97) | cens. 3/3 | cens. 3/3 |
| Centralized | 0.924 ± 0.023 | 0.971 ± 0.001 | 71 (58–84) | | |

Going from 30 to 100 rounds raised every setting (by +0.02 to +0.21), and no setting reached the
R = 100 threshold at R = 30. Even at R = 100 most validation-selected rounds are in the 80s–90s, so
many runs were still improving when they stopped. The broad pattern was the same at both lengths:
FedAdam first in every partition, FedAvg ≈ FedProx, and robust rules behind under skew. Some close
pairs swapped order (e.g. FedNova was above FedAvg under P1 at R = 30 but below it at R = 100).

### 9.6 Worst-client validation macro-F1 (validation-selected round)

| strategy | chrono P0 | chrono P1 | chrono P3 | chrono P4 | random P0 | random P1 | random P3 | random P4 |
|---|---|---|---|---|---|---|---|---|
| FedAvg | 0.772 ± 0.005 | 0.623 ± 0.149 | 0.495 ± 0.004 | 0.691 ± 0.069 | 0.853 ± 0.010 | 0.712 ± 0.025 | 0.494 ± 0.001 | 0.713 ± 0.134 |
| FedProx | 0.772 ± 0.006 | 0.623 ± 0.149 | 0.495 ± 0.004 | 0.691 ± 0.069 | 0.852 ± 0.007 | 0.712 ± 0.025 | 0.494 ± 0.001 | 0.715 ± 0.136 |
| FedNova | 0.772 ± 0.005 | 0.679 ± 0.060 | **0.646 ± 0.128** | 0.652 ± 0.013 | 0.852 ± 0.007 | 0.691 ± 0.044 | **0.560 ± 0.112** | 0.667 ± 0.105 |
| FedAdam | **0.803 ± 0.010** | **0.711 ± 0.053** | 0.509 ± 0.009 | **0.739 ± 0.076** | **0.921 ± 0.008** | **0.776 ± 0.058** | 0.496 ± 0.001 | **0.802 ± 0.084** |
| Krum | 0.777 ± 0.004 | 0.434 ± 0.220 | 0.213 ± 0.051 | 0.246 ± 0.034 | 0.845 ± 0.017 | 0.502 ± 0.286 | 0.263 ± 0.088 | 0.475 ± 0.135 |
| Median | 0.772 ± 0.006 | 0.617 ± 0.060 | 0.300 ± 0.138 | 0.590 ± 0.075 | 0.840 ± 0.006 | 0.678 ± 0.058 | 0.418 ± 0.128 | 0.560 ± 0.079 |
| Trimmed Mean | 0.774 ± 0.002 | 0.615 ± 0.052 | 0.435 ± 0.054 | 0.631 ± 0.021 | 0.840 ± 0.006 | 0.678 ± 0.048 | 0.457 ± 0.095 | 0.605 ± 0.071 |

The worst-served controller drops sharply under skew. Under P3 the worst client is at about 0.49
for FedAvg, FedProx and FedAdam; FedNova does best there (0.646 / 0.560) even though its global
score is lower. Elsewhere FedAdam gives the best worst-client score. The robust rules give the
weakest worst-client scores under every skewed partition. (Worst-client scores use each client's
own validation set, so they are in-distribution and are not affected by the chronological shift.)

### 9.7 Poisoning (chronological, R = 30, P4, f = 2 of 8 malicious)

| strategy | clean F1 | clean FNR | sign flip F1 | sign flip FNR | targeted F1 | targeted FNR |
|---|---|---|---|---|---|---|
| FedAvg | 0.468 ± 0.059 | 0.033 ± 0.010 | 0.426 ± 0.076 | 0.036 ± 0.015 | 0.419 ± 0.080 | 0.043 ± 0.014 |
| Krum | 0.250 ± 0.062 | 0.068 ± 0.118 | 0.223 ± 0.240 | 0.061 ± 0.105 | 0.365 ± 0.227 | 0.394 ± 0.476 |
| Median | 0.564 ± 0.045 | 0.087 ± 0.094 | 0.488 ± 0.081 | 0.076 ± 0.131 | 0.565 ± 0.072 | 0.200 ± 0.053 |
| Trimmed Mean | 0.565 ± 0.012 | 0.075 ± 0.086 | 0.511 ± 0.009 | 0.076 ± 0.131 | 0.518 ± 0.046 | 0.133 ± 0.114 |

- With 2 of 8 controllers malicious, sign flipping cost FedAvg 0.042 macro-F1 and targeted
  relabeling 0.049, while its attack FNR stayed below 0.05.
- Median and Trimmed Mean had higher absolute F1 than FedAvg under attack, but they also started
  higher on clean data (chronological P4), and their drop under sign flip (−0.076, −0.054) was no
  smaller than FedAvg's.
- Under targeted relabeling, every robust rule ended with a **higher** attack FNR than FedAvg
  (0.13–0.39 vs 0.04).
- Standard deviations are large (Krum ± 0.24), and the malicious set was {0, 3} for two of the
  three seeds.

### 9.8 False-positive rate [computed for this write-up from `summary.csv`; not in `paper_tables`]

At the validation-selected round, FPR (Normal flows flagged as attack):

- chronological: 0.005–0.007 for FedAvg / FedProx / FedNova / FedAdam / centralized, up to 0.024
  for Median / Trimmed Mean, and 0.288 on average for Krum (range 0–1.0);
- random R = 100: ≤ 0.015 for all except Krum (up to 0.046).

---

## 10. Statistics

All tests use **validation-selected test macro-F1 on the random protocol at R = 100**, with
**blocks = partition × seed** (4 × 3 = 12 blocks) and 7 strategies. The chronological protocol was
not tested statistically.

- **Friedman test:** χ² = 57.09, df = 6, **p = 1.75 × 10⁻¹⁰**. The strategies do not all perform
  the same.
- **Mean ranks** (1 = best): FedAdam 1.000, FedProx 2.833, FedAvg 2.958, FedNova 3.708,
  Trimmed Mean 5.333, Krum 6.000, Median 6.167. (Krum ranks ahead of Median even though its mean
  F1 is lower, 0.598 vs 0.733, because Krum does well under P0 and fails badly elsewhere.)
- **Nemenyi post-hoc**, critical difference at α = 0.05: **CD = 2.601** ranks. Significant pairs
  (p < 0.05):
  - FedAdam vs FedNova (0.035), Krum (3.0 × 10⁻⁷), Median (9.8 × 10⁻⁸), Trimmed Mean
    (1.9 × 10⁻⁵);
  - FedAvg vs Krum (0.010), Median (0.005);
  - FedProx vs Krum (0.006), Median (0.003).

  **Not** significant: FedAdam vs FedAvg (0.284), FedAdam vs FedProx (0.365), FedAvg vs Trimmed
  Mean (0.100), FedNova vs any robust rule.
- **Planned Wilcoxon signed-rank tests vs FedAvg** (two-sided, zero differences dropped, Holm
  correction over 6 comparisons; effect sizes: median difference and matched-pairs rank-biserial
  r = (W⁺ − W⁻)/(W⁺ + W⁻)):

  | strategy | better / worse / tied (of 12) | median diff | mean diff | r | p raw | p Holm | significant |
  |---|---|---|---|---|---|---|---|
  | FedProx | 4 / 3 / 5 | 0.000 | −0.001 | −0.29 | 0.578 | 0.578 | no |
  | FedNova | 3 / 9 / 0 | −0.012 | −0.021 | −0.77 | 0.016 | 0.032 | yes (worse) |
  | FedAdam | 12 / 0 / 0 | +0.054 | +0.051 | +1.00 | 0.00049 | 0.0029 | yes (better) |
  | Krum | 2 / 10 / 0 | −0.237 | −0.264 | −0.87 | 0.0049 | 0.015 | yes (worse) |
  | Median | 0 / 12 / 0 | −0.102 | −0.129 | −1.00 | 0.00049 | 0.0029 | yes (worse) |
  | Trimmed Mean | 0 / 12 / 0 | −0.090 | −0.106 | −1.00 | 0.00049 | 0.0029 | yes (worse) |

  With 12 blocks, 0.00049 is the smallest p-value the exact test can give, so FedAdam, Median and
  Trimmed Mean reached it.

- **Caveat:** blocks that share a partition scheme (same scheme, different seeds) are not fully
  independent, so the p-values are approximate. This caveat is written into the CSV.

**In plain words.**

- FedAdam beat FedAvg in every one of the 12 settings, by about 0.05 macro-F1. The all-pairs
  Nemenyi test is conservative and cannot separate the top group (FedAdam, FedProx, FedAvg).
  The planned one-against-FedAvg test can, and it finds FedAdam better.
- FedProx with μ = 0.01 is indistinguishable from FedAvg.
- FedNova is slightly but reliably worse than FedAvg (about 0.02).
- All three robust rules are clearly worse than FedAvg when no one attacks: by about 0.1 for
  Median and Trimmed Mean and about 0.25 for Krum.

The planned bootstrap confidence intervals and the regression of the gap on JSD / CV were not
done.

---

## 11. Hypotheses

Verdicts rest on the random protocol (R = 100), where differences between strategies are visible.
The chronological protocol is reported but is dominated by shift (Section 9.1).

### H1: The FedAvg–centralized gap widens as α falls, is largest under P3 and P4, and rare-class recall degrades first

**Verdict: partly supported.** The α trend was not tested, and P4 does not fit.

- Only α = 0.5 was run, so "widens as α falls" was not tested.
- Gap to centralized (0.971), FedAvg, random protocol: P0 0.088, P1 0.079, **P3 0.208**,
  P4 0.063. The gap is largest under P3, as predicted, but **P4 has the smallest gap**, smaller
  even than P0.
- The P0 gap itself (0.088) is mostly an optimization-budget effect: FedAvg was still improving
  at round 100 (selected rounds 74–99).
- Rare-class recall degrades first: BFA (fewest training rows) falls 0.77 → 0.57 → 0.00 from
  P0 → P1 → P3, while other attack classes stay ≥ 0.91. This part is supported.
- On the chronological protocol FedAvg is *above* centralized under P3 (0.634 vs 0.521), which
  cannot be interpreted as a heterogeneity effect.

### H2: FedProx, SCAFFOLD and FedAdam recover part of the gap under P3/P4 but give little benefit under IID or mild skew; FedNova's effect needs unequal τ and vanishes under P0

**Verdict: mostly not supported; the FedNova-under-P0 part is supported.**

- SCAFFOLD was not run.
- FedProx (μ = 0.01) recovered nothing: it matched FedAvg within 0.004 everywhere.
- FedAdam improved results, but the gain was **largest under IID P0** (+0.077) and P1 (+0.060),
  +0.056 under P4 and only **+0.010 under P3**. So it helped generally and did not specifically
  repair heterogeneity; the P3 gap stayed at 0.198.
- FedNova equalled FedAvg under P0 (0.883 vs 0.883; τ_k equal because shards are equal), as
  predicted. Under skew its effect was **negative** (P1 −0.017, P3 −0.017, P4 −0.048), so it
  did not recover any of the gap.

### H3: Without adversaries, Krum and Median lose accuracy relative to FedAvg under non-IID partitions; Trimmed Mean lies in between

**Verdict: supported (random protocol).**

- Under P1 / P3 / P4: Krum 0.663 / 0.222 / 0.618, Median 0.813 / 0.482 / 0.763 and Trimmed Mean
  0.822 / 0.541 / 0.784, against FedAvg 0.892 / 0.763 / 0.908.
- Trimmed Mean lies between Median and FedAvg in all three skewed partitions.
- Under IID P0 all three were close to FedAvg (0.874–0.888).
- All three are significantly worse than FedAvg in the Wilcoxon tests.
- On the chronological protocol the P3 collapse is reproduced (Krum 0.168, Median 0.264, Trimmed
  Mean 0.295 vs FedAvg 0.634), but under P4 Median and Trimmed Mean scored above FedAvg (0.564 /
  0.565 vs 0.468).

### H4: With poisoning controllers, robust rules beat FedAvg under sign flipping, but targeted attack-to-benign relabeling stays hard for every rule under P3 and P4

**Verdict: not supported in the tested setting (chronological, P4, f = 2, R = 30, 3 seeds); P3
was not tested.**

- Sign flip: Median (0.488) and Trimmed Mean (0.511) scored above FedAvg (0.426), but this
  reflects their higher clean score on chronological P4. Their losses from the attack (−0.076,
  −0.054) were no smaller than FedAvg's (−0.042). Krum was below FedAvg (0.223).
- Targeted relabeling hurt FedAvg's attack FNR very little (0.033 → 0.043) and the robust rules
  more (Median 0.087 → 0.200, Trimmed Mean 0.075 → 0.133, Krum 0.068 → 0.394 ± 0.476). It was not
  "hard for every rule": it was hard mainly for the robust rules.
- No P3 poisoning runs and no random-protocol poisoning runs were made, and the seed-to-seed
  variance is large.

---

## 12. Figures and tables

All under `results/`. Figures: 300 dpi, STIXGeneral font, Okabe-Ito colors fixed per strategy, no
titles. Tables: CSV, 3 decimals, mean and std in separate columns.

### Figures (`results/paper_figures/`)

| file | suggested caption | section |
|---|---|---|
| `fig0_system_architecture.png` | System model. Eight SDN controllers (C1–C8), each training locally on the flow data of its own switch domain (E local epochs of SGD), send model updates Δw_k to an aggregation server running one of seven strategies, and receive the global model w^(t+1). The global model is evaluated on a held-out test set never seen by clients. Two controllers are marked malicious for the poisoning study. Raw traffic never leaves a controller. | Methodology / Implementation |
| `fig_model_architecture.png` | Detection model: MLP with 68 inputs, hidden layers of 128 and 64 units (ReLU, dropout 0.2) and 5 output classes (Normal, BFA, DDoS, DoS, Probe); 17,413 parameters (68 KB in float32). | Implementation |
| `fig_pipeline.png` | Experimental pipeline as run: InSDN (3 CSVs, 343,889 flows) → preprocessing (identifier removal, feature + label deduplication, rare-family rule, 20k class cap, per-family minute split) → partitioner (P0, P1 α = 0.5, P3 m = 1, P4 α = β = 0.5; K = 8; seeds 0–2) → plain-PyTorch simulation (MLP, E = 1, B = 256, SGD lr 0.05, federated standardization) → 7 aggregation strategies → metrics and statistics. Two protocols: chronological (primary, R = 30) and random stratified 80/20 (secondary, R = 100); poisoning with f = 2 of 8 (chronological, P4). | Implementation |
| `fig1_leakage_diagnostic.png` | Leakage diagnostic: test macro-F1 and per-class recall of the centralized MLP (30 epochs, mean ± std over 3 seeds) on the chronological and the random stratified split of the same rows. The chronological split loses BFA entirely and most of Probe. | Results (first subsection) |
| `fig2_heatmap_macro_f1.png` | Validation-selected test macro-F1 (mean ± std, 3 seeds) for 7 strategies × 4 partitions: (a) random split, R = 100 (centralized 0.971 ± 0.001); (b) chronological split, R = 30 (centralized 0.521 ± 0.001). Panels use different protocols and must not be compared directly. | Results (headline) |
| `fig3_convergence_random_r100.png` | Test macro-F1 per round (mean ± std over 3 seeds), random split, R = 100, under P0 (IID, top) and P3 (attack-type exclusivity, bottom). Dashed line: centralized validation-selected macro-F1 (100 epochs). | Results (convergence) |
| `fig4_per_class_recall_random_P3.png` | Per-class test recall at the validation-selected round under P3 (m = 1), random split, R = 100, for the 7 strategies (mean ± std, 3 seeds). | Results (per-class) |
| `fig5_poisoning.png` | Poisoning on the chronological split, P4, R = 30, f = 2 of 8 malicious controllers: test macro-F1 (top) and attack false-negative rate (bottom) for FedAvg, Krum, Median and Trimmed Mean, clean vs sign flip vs targeted (attack → Normal) relabeling (mean ± std, 3 seeds). | Results (RQ3) |
| `fig6_cd_diagram.png` | Critical-difference diagram (Nemenyi, α = 0.05, CD = 2.601) of mean ranks of the 7 strategies over 12 partition × seed blocks, random split, R = 100, validation-selected test macro-F1 (Friedman p = 1.75 × 10⁻¹⁰). Strategies joined by a bar are not significantly different. | Results (statistics) |

Internal report figures (`results/report/figures/`, not styled for the paper):
`convergence_P4.png` (per-round curves under P4) and `rare_recall.png` (recall of the
smallest-training-count class per strategy and partition, both protocols).

### Tables (`results/paper_tables/`)

| file | suggested caption | section |
|---|---|---|
| `table_leakage_diagnostic.csv` | Centralized MLP, 30 epochs, 3 seeds: test macro-F1 and per-class recall on the chronological vs random split. | Results 9.1 |
| `table_valselected_macro_f1.csv` | Validation-selected test macro-F1 (mean, std, 3 seeds) per strategy and partition; chronological R = 30 and random R = 100, with centralized baselines (30 / 100 epochs). | Results 9.2 (headline) |
| `table_final_auc_macro_f1.csv` | Final-round and AUC (mean over rounds) test macro-F1 and the mean validation-selected round; chronological R = 30, random R = 100. | Results 9.2 / appendix |
| `table_per_class_recall.csv` | Per-class test recall (Normal, BFA, DDoS, DoS, Probe) at the validation-selected round; chronological R = 30, random R = 100. | Results 9.3 |
| `table_t95_bytes.csv` | Rounds and megabytes to 95% of centralized macro-F1 (T95), seeds reaching / censored; chronological R = 30 (threshold 0.495) and random R = 100 (threshold 0.922); 1.114 MB per round for all strategies. | Results 9.4 |
| `table_random_r30_vs_r100.csv` | Random split: validation-selected macro-F1 at R = 30 vs R = 100, selected round at R = 100, and seeds reaching T95 (threshold 0.922 for both). | Results 9.5 / appendix |
| `table_worst_client_val_f1.csv` | Worst-client validation macro-F1 (each client's present classes) at the validation-selected round; chronological R = 30, random R = 100. | Results 9.6 (fairness) |
| `table_poisoning.csv` | Chronological split, P4, R = 30, f = 2 of 8: test macro-F1 and attack FNR for clean, sign-flip and targeted conditions. | Results 9.7 (RQ3) |
| `table_partition_heterogeneity.csv` | Realized partition statistics per protocol, partition and seed: mean pairwise JSD, shard-size CV, min / max shard, draws, rows moved (K = 8). | Methodology / Implementation (partitioning) |
| `stats_friedman.csv` | Friedman test: 7 strategies, 12 partition × seed blocks, random R = 100, validation-selected macro-F1; χ², p and Nemenyi CD. | Results / Statistics |
| `stats_mean_ranks.csv` | Mean Friedman ranks and mean macro-F1 per strategy (random, R = 100). | Results / Statistics |
| `stats_nemenyi_pvalues.csv` | Nemenyi post-hoc p-values for all strategy pairs (random, R = 100). | Appendix |
| `stats_wilcoxon_vs_fedavg.csv` | Wilcoxon signed-rank tests of each strategy vs FedAvg over 12 blocks (random, R = 100), two-sided, Holm-corrected, with median difference and rank-biserial r. | Results / Statistics |

`results/paper_figures/fig7_flower_vs_loop.png` and `results/paper_tables/flower_validation.csv`
belong to the Flower validation and are left out of this document (see the last section).

Supporting files in `results/report/`: `tables.md` (all tables side by side) and
`t95_*.csv` (per-run T95; `t95_chrono.csv` and `t95_chrono_R30.csv` are identical;
`t95_random.csv` is the superseded R = 30 random protocol).

---

## 13. Deviations from the planned methodology (paper Section III)

| item | planned | done | reason |
|---|---|---|---|
| Datasets | InSDN, CICDDoS2019, CICIDS2017, UNSW-NB15 | InSDN only | Time-boxed study; the paper says InSDN is run first so the primary dataset is complete under any cut |
| Tasks | binary and multi-class | multi-class only | Time; partitions are built so the binary task can reuse them |
| Split | per class, earliest 80% train, ±60 s gap discarded; grouped pair split only where times are unusable | per family, cut between whole minutes, no gap; fallbacks (b) relaxed cut, (c) grouped pair split, (d) all-train | The gap rule would have discarded 28–70% of most attack families and left BFA and BOTNET without test flows (minute-resolution timestamps, few minutes per family) |
| Deduplication | "remove duplicate rows" | on features + label after identifier removal; conflicting-label groups dropped entirely; earliest kept | Duplicates that differ only in identifiers would leak across the split; conflicting labels cannot be learned |
| Identifiers | drop flow IDs, IPs, timestamps | also drop `Src Port`; keep `Dst Port` | Source port is ephemeral (shortcut risk); destination port carries service information |
| Random split | leakage diagnostic for the centralized baseline only | full secondary protocol (all strategies, R = 100); statistical tests run on it | Chronological centralized macro-F1 was only 0.521 due to shift; strategy differences were not visible there |
| Partitions | 10 settings | all 10 built (chrono, seeds 0–4); 4 run: P0, P1 α0.5, P3 m1, P4 | Time; the 4 are the IID reference plus the paper's 3 headline/tuning partitions |
| P4 | "Dirichlet α = 0.5 with quantity β = 0.5" | product reweighting q_{c,k}·s_k, normalized per family | Method not specified in the paper; this construction reduces to P1 / P2 at the limits (approved) |
| Strategies | 8 incl. SCAFFOLD | 7, no SCAFFOLD | Time |
| Framework | Flower simulation, each client a process; measured message sizes | plain PyTorch, clients sequential in one process; bytes derived | No Ray wheel on Windows + Python 3.13, and speed; Flower validation pending |
| Tuning | 12 random-search trials per strategy (lr, μ, η_s, ε) | none; lr 0.05, μ 0.01, η_s 0.01, ε 1e-3 fixed | Time |
| K, E | K ∈ {4, 8, 16}, E ∈ {1, 5} sweeps | K = 8, E = 1 only | Time |
| Rounds | R = 100 | chrono R = 30; random R = 100 (R = 30 kept for comparison) | Time; R = 30 proved too short on the random protocol, so it was extended |
| Seeds | 5, or 10 for headline configurations | 3 (0–2) | Time |
| Baselines | centralized, local-only, FedAvg-IID; same tuning budget | centralized (epochs = rounds, 10% stratified validation) and FedAvg P0; no local-only | Time |
| Model | MLP + 1D-CNN | MLP only (17,413 params) | Time |
| Round selection | not specified (tuning used pooled validation) | headline = test score at the best pooled-validation round; final and AUC also reported | Avoids reporting an arbitrary last round; test never used for selection |
| Rare-class recall | mean recall over the lowest-quartile classes | recall of the single class with the fewest training rows (DDoS chrono, BFA random) | With 5 classes the lowest quartile is one class |
| Fairness | std and minimum of per-client F1, number of present classes | minimum (worst-client) only | Not implemented |
| Precision, CPU time, server time | reported | precision not reported; per-round wall time only | Not implemented; sequential simulation makes per-client CPU time and server replay not comparable to the planned setup |
| T95 | restricted mean min(T95, R), fraction reaching | mean over seeds reaching + count censored | Simpler with 3 seeds |
| Poisoning | K ∈ {8, 16}, f ∈ {0, 1, 2}, P0 and P4, InSDN + CICIDS2017, three attacks | K = 8, f = 2, P4, chronological R = 30, sign flip + targeted | Time; random label flip not run |
| Statistics | Friedman/Nemenyi over dataset × task × partition blocks (80); within-configuration Wilcoxon on 10 seeds; bootstrap CIs; regression on JSD/CV | Friedman/Nemenyi and Wilcoxon + Holm over 12 partition × seed blocks (random, R = 100) | Only one dataset / task and 3 seeds; bootstrap and regression not done |
| Emulated testbed (Mininet + ONOS, T-SDN) and deployment-mode timing | planned | not done | Time |
| Resumability | `ckpt.pt` per round with RNG states | per-run completion marker (JSON); interrupted runs restart from round 1 | Runs take 20–120 s, so per-round checkpoints were unnecessary |

---

## 14. Limitations, threats to validity and future work

### Limitations and threats to validity

- **Scope:** one dataset, one task, one model, K = 8, E = 1, 3 seeds and 4 partitions. The
  results may not generalize to other datasets, feature spaces or larger federations.
- **No tuning:** one client learning rate (0.05) for every strategy, and fixed FedProx μ = 0.01 and
  FedAdam η_s = 0.01, ε = 1e-3. FedProx with this μ is effectively FedAvg, so the results say
  nothing about FedProx at larger μ. FedAdam's advantage may partly reflect a better effective
  step size than a tuned FedAvg would have.
- **Chronological test set:**
  - BFA's test set is a single address pair, unseen in training;
  - DoS has only 10.8% test (the only acceptable cut) and DDoS 42.7%;
  - centralized macro-F1 is 0.521, so strategy comparisons on this protocol are dominated by
    shift and are not reliable.
- **Random protocol leakage:** flows from the same sessions fall on both sides of the split, so
  scores are inflated. The statistical ranking rests on this protocol, and the ranking without
  leakage could differ.
- **Training budget:** even at R = 100 most validation-selected rounds were in the 80s–90s, and
  FedNova, Krum, Median and Trimmed Mean never reached T95 in any partition, so the runs had not
  fully converged.
  Some gaps (including FedAvg P0 vs centralized) are partly a budget effect.
- **Rare classes:** BOTNET, Web-Attack and U2R (364 flows) are excluded from the multi-class task.
  Under P3, one or two clients per seed hold only Normal + Rare and therefore train on Normal only.
- **Statistics:** 12 blocks that share partition schemes are not independent, so p-values are
  approximate. There are no bootstrap CIs, no tests on the chronological protocol, and no
  heterogeneity regression.
- **Poisoning:** one setting (chronological P4, R = 30, f = 2), 3 seeds, with the same malicious
  set {0, 3} in two of them, and large variance. No P3 poisoning, so the "sole holder of a family"
  scenario of H4 was not tested.
- **Cost measures:** bytes are derived, not measured. Wall time comes from 4 processes on 2
  physical cores, and client CPU time and server aggregation time were not measured.
- **Implementation:** a sequential plain-PyTorch simulation, not Flower; equivalence with Flower
  is the subject of the pending validation below.

### Future work

- Run the remaining planned parts: the binary task, the other three datasets, all 10 partitions
  (including the P1 α sweep and P2), SCAFFOLD, 5–10 seeds, the K and E sweeps, local-only
  baselines and the 1D-CNN.
- Apply the equal-budget tuning protocol, especially FedProx μ and the client learning rate per
  strategy.
- Revisit the chronological split for InSDN (e.g. families with more address pairs or more
  distinct minutes) or report per-class results that exclude shift-limited classes; increase R on
  the chronological protocol.
- Poisoning under P3 and P0, f ∈ {1, 2}, random label flip, and on the random protocol.
- Measure client CPU time, server aggregation time and serialized message sizes (Flower), and
  run the Mininet / ONOS transfer check.
- Bootstrap confidence intervals and the regression of the gap to centralized on realized JSD and
  CV.

---

## Flower validation: pending
