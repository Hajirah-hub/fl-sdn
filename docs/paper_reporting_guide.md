# What to report in the paper: experiment details, results, figures and tables

**Purpose.** `docs/paper.pdf` currently ends at Section III (methodology, written as a plan). This
guide says what to put in the sections that follow — the experimental setup and the results /
discussion — and which parts of Section III must be rewritten to describe what was actually done.
It picks the evidence that makes the case strong **and** survives review, and names the exact
figure / table file for each item. Every number here comes from `docs/experiment_writeup.md`
(the full record) and `results/paper_tables/`.

File locations: figures in `results/paper_figures/`, tables in `results/paper_tables/`.

---

## 1. The story the evidence supports

State it in the abstract, introduction contributions and conclusion in these terms — each part is
backed by a statistic, not an impression:

1. **Where the aggregation rule matters, adaptive server optimization wins.** FedAdam beats FedAvg
   in **20 of 20** partition × seed blocks, +0.034 macro-F1 (bootstrap 95% CI [0.024, 0.046];
   Wilcoxon, Holm p = 1.3 × 10⁻⁵), and still beats FedAvg / FedProx run at a 2–3× larger learning
   rate in 17 of 20 blocks.
2. **Controller-level heterogeneity is costly mainly through label skew and attack-type
   exclusivity, not through unequal data volume.** The gap to centralized training rises with
   realized JSD for 6–7 of 8 strategies; shard-size CV never matters. Attack-type exclusivity (P3)
   is the hardest setting for every rule, and the rarest attack family (BFA) disappears first.
3. **Byzantine-robust rules are a bad default for heterogeneous SDN controllers.** Without any
   attacker, Krum, Median and Trimmed Mean lose 0.24 / 0.10 / 0.09 macro-F1 to FedAvg (all
   significant), and under attack they still do not beat FedAvg. When a malicious controller is
   the only holder of an attack family, no rule can recover it — the core RQ3/H4 point.
4. **Federation is clearly worth it under skew:** a controller training alone reaches 0.21
   macro-F1 under P3 vs 0.78 (FedAvg) and 0.84 (FedAdam).
5. **Honest negative result:** on the leakage-aware chronological split, distribution shift
   (unseen time periods / address pairs) dominates — centralized training reaches only 0.52 and
   most rules sit at the same level. This is itself a finding for FL-IDS evaluation: high
   benchmark accuracies should be read with caution, because on InSDN a leakage-aware split cut
   centralized macro-F1 from 0.96 (random split) to 0.52 for the same rows, model and budget.

The paper explicitly allows result 5 and "FedAvg stays competitive" outcomes (end of Sec. I-B), so
reporting them strengthens rather than weakens the paper.

---

## 2. Section III must be updated to describe what was run

The current Section III is written in the future tense as a plan for 4 datasets and ~8,000 runs.
A reviewer will compare it with the results, so either rewrite it in the past tense for what was
done or add a "Deviations" paragraph. **Every row below must be stated somewhere.** (Full table:
`experiment_writeup.md` Section 13.)

| paper says | what was done — say this | why it is defensible |
|---|---|---|
| 4 datasets, binary + multi-class | **InSDN only, multi-class only** | Paper already says InSDN is run first and is the primary dataset (III-J). Present the other three as future work. |
| Earliest 80% per class, ±60 s gap discarded | **Per-family split between whole minutes (minute cut, test share 10–35%, closest to 20%, ≥ 100 test flows), with fallbacks: relaxed cut ≤ 50%, grouped address-pair split, all-train** | InSDN timestamps have minute resolution and few distinct minutes per family; the gap rule would discard 28–70% of most attack families and leave BFA and BOTNET with no test flows. Give the per-family method table (DoS 10.8%, DDoS 42.7% relaxed because sources are spoofed, BFA grouped by address pair). |
| Random split only as a leakage diagnostic | **Random stratified 80/20 split used as a second full protocol; statistics and tuning run on it** | The chronological split leaves almost no between-strategy signal (Sec. 4.1 below). Say clearly: random-split scores are inflated by leakage and are used **only to rank strategies against each other**, never compared with chronological scores or with other papers. |
| Flower simulation, measured message sizes | **Plain PyTorch simulation (clients sequential in one process); bytes derived** | Validated against a real Flower run: per-round macro-F1 within 0.0012, measured messages 1.028 × derived size (Fig. Flower, Sec. 4.7 below). |
| 10 partition settings, 5/10 seeds | **Main grid P0, P1 α=0.5, P3 m=1, P4 (the paper's headline + IID); sweep P1 α ∈ {0.1, 1, 10}, P2 β=0.5 for FedAvg / FedAdam; 5 seeds (0–4)** | Covers all three headline configurations of III-J plus the IID reference. |
| P4 "Dirichlet α=0.5 with quantity β=0.5" | **Product reweighting: family c dealt by q_{c,k}·s_k / Σ_j q_{c,j}·s_j** | The paper does not define how the two are combined; this one reduces exactly to P1 (uniform s) and P2 (uniform q), unit-tested. |
| R = 100 everywhere | **Random protocol R = 100; chronological R = 30** | Time; the chronological protocol is shift-dominated, so longer training was not expected to change its conclusions. |
| K ∈ {4, 8, 16}, E ∈ {1, 5}, 1D-CNN, testbed, deployment-mode timing | **K = 8, E = 1, MLP only; no testbed, no CPU / server timing** | Out of time; list as limitations / future work. Do not keep "client CPU time" in the metric list. |
| Tuning on "a seed reserved for tuning" | **Seed 99 with its own partitions, random protocol, R = 30**; the 12 client lrs drawn once and shared; centralized baseline tuned the same way | Exactly the paper's protocol (III-H), with the protocol choice stated. |
| Rare-class recall = mean over lowest-quartile classes | **Recall of the single class with the fewest training rows** (BFA random; DDoS chronological) | With 5 classes the lowest quartile is one class. |
| T95: restricted mean min(T95, R) | **Mean over seeds that reach T95 + number censored, for two centralized references** | Needed because some centralized runs collapsed (Sec. 4.6). |
| Statistics: within-configuration Wilcoxon on 10 seeds; Friedman over dataset × task × partition blocks (80) | **Friedman / Nemenyi, Wilcoxon + Holm, bootstrap CIs and JSD/CV regression over 20 partition × seed blocks** | Only one dataset / task exists. State the dependence caveat (Sec. 5). |
| Model selection: unspecified | **Headline = test score at the round with the best pooled validation macro-F1** (test never used) | Prevents reporting an arbitrary last round; final round and AUC also reported. |
| Fairness: std + min of per-controller F1 | **Minimum (worst-controller) validation F1 only** | Say so. |
| Poisoning on P0 and P4, f ∈ {0,1,2}, K ∈ {8,16}, 3 attacks | **f = 2 of 8; chronological P4 (R=30) and random P3 (R=100); sign flip and targeted relabeling** | P3 added on purpose to test H4's "only holder of an attack family" case. Random label flipping not run. |
| SCAFFOLD per the original paper | **Option II, global step 1, control variates aggregated with n_k weights** | Makes "SCAFFOLD with zero variates == FedAvg" exact (the unit test the paper promises). |
| Table IV: 710 core-hours estimated | **834 runs, 11.2 process-hours, about 2.9 h wall-clock on a 2-core / 4-thread laptop CPU (4 workers)** | Replace the estimate with the actual cost — it strengthens the "feasible on student hardware" claim. |

---

## 3. Experimental setup section (new Section IV) — what to note

Keep it short; point to the tables.

### 3.1 Implementation and environment
- Python 3.13.18, PyTorch 2.15.0 (CPU), Flower 1.43.0 (validation only), Intel i7-7600U
  (2 cores / 4 threads), 7 GB RAM, WSL 2. Code, partition files, config and lock files released.
- Plain-PyTorch federated loop; each run deterministic given seed (one seed controls partition,
  initialization, dropout; batch order per (seed, round, client)).
- 53 unit tests, including **SCAFFOLD with zero variates equals FedAvg (bitwise)**, **FedNova
  equals FedAvg with equal τ**, **federated standardization equals pooled standardization**.
  Reviewers like these; mention them in one sentence.

### 3.2 Dataset after preprocessing (→ **Table: dataset counts**)
- InSDN: 343,889 raw flows → 218,685 after deduplication on features + label (conflicting-label
  groups removed) → rare families BOTNET 164 + Web-Attack 183 + U2R 17 = 364 < 500 excluded from
  the multi-class task → 20,000-flow class cap → **68 features**, 5 classes.
- Table rows (train / test): chronological Normal 15,987 / 4,013; BFA 1,087 / 294; DDoS 871 / 649;
  DoS 17,836 / 2,164; Probe 16,166 / 3,834 (total 51,947 / 10,954). Random protocol: 50,321 /
  12,580 (Normal, DoS, Probe 16,000 / 4,000; BFA 1,105 / 276; DDoS 1,216 / 304).
- Per-family split method (one small table or a sentence): Normal, DoS (10.8%), Probe → minute
  cut; DDoS → relaxed cut 42.7% (spoofed sources); BFA → grouped address pair (3 pairs; one held
  out); BOTNET → all train.
- Source: `experiment_writeup.md` Section 3.

### 3.3 Controller-level partitions (→ **Table: realized heterogeneity**)
- K = 8, 10% local validation per controller, n_min = 100 met in ≤ 3 draws, no rows moved.
- Report realized **JSD and CV ranges** per setting, not only nominal α/β — the paper promised
  this ("performance analysed against realized statistics"). P0 0/0; P1 α0.5 JSD 0.33–0.43;
  P3 0.48–0.53; P4 0.22–0.40 with CV 0.82–1.29; P1 α0.1 0.66–0.74; P2 CV 0.94–1.39 with JSD ≈ 0.
- One sentence on P3 layout: 5 attack families over 8 controllers → each family has 1–2 holders;
  some controllers hold Normal only.
- Source: `table_partition_heterogeneity.csv` (condense to ranges).

### 3.4 Model, training, baselines
- MLP 68 → 128 → 64 → 5, ReLU, dropout 0.2, **17,413 parameters (68 KB)** → message 1.11 MB per
  round for K = 8 (2.23 MB for SCAFFOLD). Use `fig_model_architecture.png` only if space allows.
- Plain SGD, no momentum, batch 256, E = 1, class-weighted cross-entropy with global counts,
  federated standardization from counts / sums / sums of squares.
- Baselines: centralized (10% stratified validation, epochs = rounds), local-only (each controller
  alone, 100 epochs, scored on the global test set), FedAvg on P0.

### 3.5 Hyperparameter tuning (→ **Table: frozen hyperparameters**, Fig. tuning optional)
This is a strong point — the paper's "equal-budget tuning" promise is fully met. Report:

| strategy | client lr | other | best val. score | median | range over 12 trials |
|---|---|---|---|---|---|
| Centralized | 0.202 | — | 0.919 | 0.847 | 0.738–0.919 |
| FedAvg | 0.256 | — | 0.848 | 0.762 | 0.593–0.848 |
| FedProx | 0.202 | μ = 0.0012 | 0.842 | 0.760 | 0.593–0.842 |
| FedNova | 0.256 | — | 0.843 | 0.747 | 0.626–0.843 |
| FedAdam | 0.0144 | η_s = 0.285, ε = 0.0067 | 0.878 | 0.793 | 0.436–0.878 |
| SCAFFOLD | 0.202 | — | 0.880 | 0.780 | 0.626–0.880 |
| Krum | 0.202 | — | 0.402 | 0.366 | 0.179–0.402 |
| Median | 0.256 | — | 0.594 | 0.543 | 0.478–0.594 |
| Trimmed Mean | 0.256 | — | 0.613 | 0.540 | 0.508–0.613 |

- The score spread across trials is what the paper promised as its tuning sensitivity measure
  (threat (iii), III-N): FedAdam has the widest spread (0.44–0.88), i.e. it is the hardest to tune.
  Say this — it pre-empts "FedAdam won because it got more tuning".
- Source: `table_tuning_chosen.csv`, `table_tuning_trials.csv`, `fig8_tuning.png`.

### 3.6 Evaluation protocol and metrics
- Two protocols (chronological primary, random secondary) — one paragraph, with the leakage
  caveat.
- Headline metric: **validation-selected test macro-F1** (mean ± std over 5 seeds). Also per-class
  recall, worst-controller validation F1, attack FNR, FPR, T95 and MB-to-T95.
- Failure flags: diverged (non-finite weights) and collapsed (one class predicted for all test
  flows at the final round).

### 3.7 Actual compute (replaces Table IV)
834 runs (300 tuning, 165 random grid, 205 chronological incl. 40 poisoning, 40 sweep, 20
local-only, 40 random poisoning, 24 + 40 learning-rate checks), 11.2 process-hours,
**≈ 2.9 h wall-clock** with 4 worker processes on one laptop CPU.

---

## 4. Results section (new Section V) — order, numbers, figures

Recommended order: protocol check first (it explains why the random protocol carries the
comparisons), then RQ1, RQ2, RQ3, then stability and implementation validation, then a hypothesis
summary.

### 4.1 Leakage-aware vs random split (→ **Fig.: leakage diagnostic**, `fig1_leakage_diagnostic.png`)
- Centralized, 30 epochs, same rows, stable seeds: chronological **0.522 ± 0.011** (seeds 0, 1, 3)
  vs random **0.956 ± 0.014** (seeds 0–3) [all 5 seeds: 0.519 ± 0.009 vs 0.926 ± 0.070]. BFA
  recall **0.005** vs 0.805; Probe 0.31 vs 0.94 (all-seed means).
- Cause, already diagnosed: not underfitting (training F1 0.88), not undertraining (100 epochs
  still 0.52). BFA's test set is an address pair never seen in training.
- Key sentence: *with a leakage-aware split, most of the error comes from shift, not from the
  aggregation rule; strategy comparisons are therefore made on the random split and only relative
  rankings are interpreted.*
- Chronological headline (Table, panel b of the heatmap): 0.51–0.53 for almost all rules in P0,
  P1, P4; only P3 separates them (FedAvg 0.675, robust rules 0.11–0.28).

### 4.2 RQ1 — cost of controller-level heterogeneity
Main numbers (random, R = 100, stable centralized 0.972 ± 0.001):

- FedAvg: P0 **0.956**, P1 0.938, **P3 0.778**, P4 0.941 → gap 0.016 / 0.034 / **0.194** / 0.031.
- Rare class first: FedAvg BFA recall 0.79 (P0) → 0.75 (P1) → **0.00 (P3)** while DoS / Probe /
  DDoS stay ≥ 0.95 (`fig4_per_class_recall_random_P3.png`).
- α sweep: FedAvg 0.956 (IID), 0.934 (α10), 0.952 (α1), 0.938 (α0.5), 0.923 (α0.1) — only extreme
  Dirichlet skew costs much, and P3 at JSD ≈ 0.5 costs far more than α = 0.1 at JSD 0.69, because
  exclusivity removes whole families from most controllers (`fig9_heterogeneity_sweep.png`).
- Regression: label skew (JSD) predicts the gap, quantity skew (CV) does not (no CV coefficient
  significant) (`fig10_gap_regression.png`, `stats_regression_gap.csv`). This is the
  "disentangling heterogeneity" analysis the paper promised (III-M).
- Local-only vs federated (`fig12_local_only.png`): P0 0.950 vs 0.956 (federation adds little);
  **P3 0.214 (best controller 0.320) vs 0.778 / 0.842**; worst controller under P1 / P4 0.45–0.50.
  This is the "value of collaboration" the local-only baseline was planned for.
- Worst-controller validation F1 under P3: FedAvg 0.50 (the BFA holder) — fairness evidence.

### 4.3 RQ2 — which strategies recover the loss, at what cost
Headline (→ **Table: validation-selected macro-F1**, **Fig.: heatmap** `fig2_heatmap_macro_f1.png`):

| random, R = 100 | P0 | P1 | P3 | P4 |
|---|---|---|---|---|
| FedAvg | 0.956 | 0.938 | 0.778 | 0.941 |
| FedProx | 0.957 | 0.929 | 0.771 | 0.946 |
| FedNova | 0.951 | 0.943 | 0.822 | 0.938 |
| **FedAdam** | **0.967** | **0.971** | 0.842 | **0.969** |
| SCAFFOLD | 0.849 | 0.900 | **0.871** | 0.945 |

Statistics (→ **Table: Wilcoxon + CIs**, **Fig.: CD diagram** and/or **bootstrap CI plot**):
- Friedman χ² = 64.83, p = 1.6 × 10⁻¹¹ (8 strategies, 20 blocks); Nemenyi CD = 2.35; mean ranks
  FedAdam 1.40, FedAvg 3.73, FedNova 3.92, SCAFFOLD 4.25, FedProx 4.30, Trimmed Mean 5.55,
  Median 6.05, Krum 6.80 (`fig6_cd_diagram.png`).
- vs FedAvg: FedAdam +0.034 [0.024, 0.046], 20/0, Holm p = 1.3 × 10⁻⁵; FedProx −0.003, FedNova
  +0.010, SCAFFOLD −0.012 (all n.s.) (`fig11_bootstrap_ci_vs_fedavg.png`,
  `stats_wilcoxon_vs_fedavg.csv`, `stats_bootstrap_ci_vs_fedavg.csv`).
- Where each method helps: FedAdam everywhere, most under P3 (+0.064); **SCAFFOLD and FedNova help
  specifically under P3** (+0.093 / +0.044, keeping part of BFA: recall 0.38 / 0.34 vs 0.00) —
  consistent with H2's "recover the gap under exclusivity". FedProx with tuned μ = 0.0012 is
  FedAvg.
- Cost (→ `table_t95_bytes.csv`): with the stable threshold, **only FedAdam reaches 95% of
  centralized in every seed of P0, P1 and P4** (32–57 rounds, 35–64 MB); nobody reaches it under
  P3 except SCAFFOLD in 2 of 5 seeds, and SCAFFOLD's doubled messages make it 2× as expensive per
  round. Convergence curves: `fig3_convergence_random_r100.png` (appendix).
- Learning-rate sensitivity (one paragraph — it pre-empts the obvious reviewer question): the
  tuned client lrs sit near the top of the search range, so FedAvg / FedProx were rerun at lr 0.6
  on the full grid. Not better overall (−0.006 / +0.006, n.s.), better under P3 (≈ 0.86) but worse
  under P0 / P1 with 4 divergences, and **FedAdam still wins 17 of 20 blocks** (+0.040 / +0.030,
  CIs exclude zero). Source: `stats_lr_sensitivity.csv`, `table_lr_sensitivity.csv`.

### 4.4 RQ3a — robust rules with honest but heterogeneous controllers (H3)
- Random: Krum 0.816 / **0.143** / 0.735, Median 0.864 / 0.535 / 0.846, Trimmed Mean 0.885 /
  0.570 / 0.849 under P1 / P3 / P4, vs FedAvg 0.938 / 0.778 / 0.941; all equal FedAvg under P0
  (0.956–0.957).
- All significantly worse than FedAvg (Holm p ≤ 0.0011; CIs exclude zero), and their gap grows
  fastest with JSD (slopes 0.6–1.4 vs 0.12 for FedAvg).
- Mechanism (per-class recall): under P3 Krum follows one controller and loses DoS and Probe
  entirely; Median / Trimmed Mean lose most of DoS — they discard exactly the minority-controller
  updates H3 predicted.
- Reproduced on the chronological split (P3: Krum 0.107, Median 0.275, Trimmed Mean 0.230 vs
  FedAvg 0.675) — **the one strategy effect that survives the shift**, so state it.

### 4.5 RQ3b — poisoning (→ **Fig.: poisoning** `fig5_poisoning.png`, **Table: poisoning**)
- Random P3, f = 2 of 8: sign flip drops FedAvg 0.778 → **0.497** (2 of 5 runs diverged); robust
  rules attacked: Krum 0.158, Median 0.416, Trimmed Mean 0.456 — **still below FedAvg**, because
  their honest baseline is already low.
- Targeted relabeling raises attack FNR for every rule under P3 (FedAvg 0.002 → 0.206, Median
  0.098 → 0.253, Trimmed Mean → 0.196, Krum → 1.0) — H4's second half confirmed.
- **Sole-holder effect** (the key RQ3 result, `table_poisoning_random_P3.csv`): in the 3 seeds
  where a malicious controller is the only holder of an attack family (DoS, Probe, BFA), FedAvg
  sign flip 0.371 vs 0.688 otherwise; targeted 0.615 vs 0.765; same pattern for Median / Trimmed
  Mean. A majority-based defence cannot restore information that only the attacker held.
- Chronological P4: attacks barely move FedAvg (0.519 clean, 0.527 sign flip, 0.485 targeted);
  Krum is the most damaged (targeted FNR 0.58).

### 4.6 Stability at tuned learning rates (short subsection or paragraph; **Table: divergence /
collapse** in appendix)
Report it — it is a genuine, useful finding and hiding it would be the weakest point in review.
- SCAFFOLD diverged in **25%** of random and **45%** of chronological runs at its tuned lr
  (selected on the tuning seed, where the next larger lr already diverged).
- Centralized random seed 4 collapsed after epoch 3; results are reported with all 5 seeds
  (0.938 ± 0.075) and with the 4 stable ones (0.972 ± 0.001); T95 and the gap regression are shown
  both ways and the conclusions do not change.
- Practical message: in cross-silo SDN FL, stability under the chosen learning rate should be part
  of strategy selection; FedAdam was stable in every run.
- Source: `table_divergence_collapse.csv`.

### 4.7 Implementation validation (one paragraph + appendix figure `fig7_flower_vs_loop.png`)
Plain loop vs real Flower simulation (built-in FedAvg / FedMedian, custom FedAdam), random P4,
seed 0, R = 30: max per-round |Δ macro-F1| ≤ 0.0012; measured messages 1.028 × the derived size;
Flower ~2× slower per round. This defends both the framework deviation and the bytes metric.
(Run with the earlier untuned settings; say so.)

### 4.8 Hypothesis summary (→ **Table: hypotheses and verdicts**)

| hypothesis | verdict | evidence in one line |
|---|---|---|
| H1 gap grows as α falls, largest under P3/P4, rare class first | **Partly supported** | Largest under P3 (0.194); P4 no harder than P1; α trend weak, non-monotone; JSD slope significant; BFA lost first |
| H2 FedProx / SCAFFOLD / FedAdam recover gap under P3/P4, little under IID; FedNova only with unequal τ | **Partly supported** | FedAdam gain smallest under IID (+0.011), largest under P3 (+0.064); SCAFFOLD +0.093 and FedNova +0.044 under P3; FedProx (μ ≈ 0.001) no effect; SCAFFOLD unstable under P0/P1 |
| H3 Krum / Median lose under non-IID honest controllers, Trimmed Mean in between | **Supported** | −0.24 / −0.10 / −0.09, all significant; equal to FedAvg only under IID |
| H4 robust rules beat FedAvg under sign flip; targeted relabeling hard for all under P3/P4 | **First part not supported in the tested settings; second supported under P3** | Robust rules never overtake FedAvg; targeted raises FNR for every rule under P3; sole-holder runs worst |

---

## 5. Making the case defensible — likely reviewer questions and the answer to give

| reviewer question | answer (with the evidence) |
|---|---|
| "Your main comparison uses a random split you yourselves call leaky." | Yes, deliberately and only for ranking: the leakage-aware split was run in full, and the diagnostic shows it is shift-dominated (centralized 0.52, BFA recall 0). Rankings, not absolute scores, are interpreted on the random split, and the robust-rule collapse under P3 reproduces on the chronological split. |
| "FedAdam just got a better learning rate." | Every strategy got the same 12-trial budget with shared client-lr draws; FedAdam's score spread was the widest (hardest to tune); and it still beats FedAvg / FedProx run at lr 0.6 in 17 of 20 blocks. |
| "The tuned lr hits the search boundary." | Checked on the tuning seed (lr 0.4 / 0.6 score higher for FedAvg / FedNova / FedProx) and on the full grid (lr 0.6 not better overall; P3 improves, IID worsens). Report it as a limitation, not hidden. |
| "20 blocks from 4 partitions are not independent." | Agreed; p-values are labelled approximate, the bootstrap resamples seeds within partitions, and effect sizes are reported. FedAdam's 20-of-20 wins and the robust rules' 16–17-of-20 losses are large and consistent effects, not borderline p-values. |
| "SCAFFOLD results look odd." | SCAFFOLD diverged in 25–45% of runs at its tuned lr; this is reported with exact counts; its P3 advantage (0.871) comes from non-diverged runs as well. |
| "Only one dataset." | InSDN is the paper's designated primary dataset (III-J) and the only SDN-generated one; other datasets are future work. Say this in the abstract and the limitations — do not let the abstract still claim four datasets. |
| "No real Flower deployment." | Validated equivalence (≤ 0.0012 per round) and message size (1.028 ×) in a real Flower simulation. |
| "Is the class weighting / standardization leaking pooled data?" | No — both are computed from per-controller counts, sums and sums of squares; unit-tested equal to pooled statistics. |

**Claims to avoid** (they will not survive review):
- Any absolute random-split score compared with related work ("we reach 0.97 like [14]").
- "FedAdam fixes heterogeneity" — its gain is general (also under IID), largest under P3 but it
  still leaves most of the P3 gap.
- "SCAFFOLD is the best under exclusivity" without the instability caveat; and P3 rankings without
  noting they depend on the learning rate (FedAvg at lr 0.6 reaches 0.859 under P3).
- "Robust rules protect against poisoning" — not in this setting.
- Anything about CPU / server time, K / E effects, binary task, other datasets, the testbed or the
  1D-CNN — not measured.
- Bytes as an independent result — they are derived (rounds × message size), as the paper itself
  says in III-L.

---

## 6. Recommended figures and tables for the paper

IEEE conference papers usually fit 6–8 figures and 5–6 tables. Suggested selection; everything
else goes to an appendix or the released artifact.

### Main text — figures

| # | file | role | priority |
|---|---|---|---|
| Fig. 1 | `fig_pipeline.png` (or redraw the existing Fig. 1 to match it) | Pipeline **as run**: InSDN, minute split, P0–P4, plain loop + Flower check, 8 strategies, two protocols. The current Fig. 1 still shows 4 datasets, the testbed and CPU time — replace it. | required |
| Fig. 2 | `fig0_system_architecture.png` | System model: 8 controllers, server, malicious controllers, raw traffic never leaves a controller | high |
| Fig. 3 | `fig1_leakage_diagnostic.png` | Why the random protocol is used; shift diagnosis | required |
| Fig. 4 | `fig2_heatmap_macro_f1.png` (double column) | Headline results, both protocols, 8 × 4 | required |
| Fig. 5 | `fig6_cd_diagram.png` **or** `fig11_bootstrap_ci_vs_fedavg.png` | Statistical summary (CD diagram for all pairs; CI plot for "vs FedAvg" with effect sizes) — pick one, the CI plot is easier to read | required |
| Fig. 6 | `fig4_per_class_recall_random_P3.png` | Rare family lost under exclusivity; why SCAFFOLD / FedNova / FedAdam help | high |
| Fig. 7 | `fig12_local_only.png` | Value of federation | high |
| Fig. 8 | `fig5_poisoning.png` (double column) | RQ3 / H4 | required |

### Appendix / supplementary figures
`fig3_convergence_random_r100.png` (convergence and SCAFFOLD's divergence), `fig9_heterogeneity_sweep.png`
(α sweep; promote to main text if there is space — it answers H1 directly),
`fig10_gap_regression.png` (JSD vs CV), `fig8_tuning.png` (tuning spread),
`fig7_flower_vs_loop.png` (implementation validation), `fig_model_architecture.png`.

### Main text — tables

| # | content | source |
|---|---|---|
| Table I | Strategies (existing) — add a column with the **tuned values** | existing + `table_tuning_chosen.csv` |
| Table II | Search spaces (existing) — add best / median / range of tuning scores | `table_tuning_chosen.csv` |
| Table III | Factors **as run** (replace levels: InSDN, multi-class, P0/P1/P3/P4 + sweep, K=8, E=1, R=100/30, f=2, seeds 0–4) | Section 2 of this guide |
| Table IV | Dataset counts after preprocessing (train / test per class, both protocols) and actual compute (replace the estimated budget) | Sections 3.2, 3.7 |
| Table V | Validation-selected macro-F1, 8 strategies × 4 partitions, random R=100 (+ chronological, or chronological in appendix) with centralized | `table_valselected_macro_f1.csv` |
| Table VI | Statistics vs FedAvg: wins/losses, mean diff, 95% CI, Holm p, rank-biserial r (+ Friedman line) | `stats_wilcoxon_vs_fedavg.csv`, `stats_bootstrap_ci_vs_fedavg.csv`, `stats_friedman.csv` |
| Table VII | Poisoning: clean / sign flip / targeted, F1 and FNR, chrono P4 and random P3 (with sole-holder column) | `table_poisoning.csv`, `table_poisoning_random_P3.csv` |
| Table VIII | Hypotheses and verdicts | Section 4.8 of this guide |

### Appendix tables
Realized heterogeneity (`table_partition_heterogeneity.csv`, condensed to ranges), T95 and MB
(`table_t95_bytes.csv`), per-class recall (`table_per_class_recall.csv`), worst-controller F1
(`table_worst_client_val_f1.csv`), α sweep (`table_alpha_sweep.csv`), local-only
(`table_local_only.csv`), divergence / collapse (`table_divergence_collapse.csv`), learning-rate
boundary and sensitivity (`table_lr_boundary.csv`, `table_lr_sensitivity.csv`,
`stats_lr_sensitivity.csv`), JSD / CV regression (`stats_regression_gap.csv`), Flower validation
(`flower_validation.csv`).

---

## 7. Limitations paragraph — what must be in it

1. One dataset (InSDN), multi-class task, MLP, K = 8, E = 1, 5 seeds.
2. Chronological split dominated by shift; strategy rankings rest on the random split, whose
   absolute scores are inflated by leakage.
3. Tuned learning rates at the edge of the search range; tuning at R = 30 on the random protocol.
4. Instability at tuned rates (SCAFFOLD 25–45% divergence; one collapsed centralized seed).
5. Statistical blocks share partition schemes; p-values approximate.
6. Bytes derived (validated 1.028 ×); no CPU / server / deployment timing; no testbed transfer
   check.
7. Poisoning limited to f = 2, two attacks, two settings; sole-holder split rests on 3 vs 2 seeds.
8. The literature search protocol of Section II-A must actually be executed and its date and
   counts recorded before submission (the paper says so itself).
