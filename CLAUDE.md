# CLAUDE.md — Federated aggregation study on multi-controller SDN

Single-person implementation of Section III of `docs/paper.pdf` (methodology only).
Primary dataset: InSDN. Other datasets (CICDDoS2019, CICIDS2017, UNSW-NB15) follow the same
pipeline later; each dataset is an independent federation — never merge datasets.

## Environment

- Python **3.13** (Kaggle runs 3.13.18; local venv uses 3.13.17 because 3.13.18 has no Windows
  installer — same `cp313` ABI, same wheels). PyTorch **CPU** build. Flower in simulation mode.
- Exact versions: `requirements.txt` (direct pins) and `requirements-lock.txt` (full freeze).
  Never upgrade a pin casually — Flower's simulation API changes between releases.
- **Two local environments:**
  - **Windows `.venv/`** (Python 3.13.17) is for **unit tests only**:
    `.venv\Scripts\activate` → `pip install -r requirements.txt`. There is no Ray wheel for
    Windows + Python 3.13, so `flwr[simulation]` installs without Ray and simulations cannot run there.
    Keep strategy and client logic testable without Ray (call them in-process in tests).
  - **WSL (Ubuntu 26.04, distro installed at `D:\WSL\Ubuntu`) `~/venvs/fedsdn`** (Python 3.13.18,
    same as Kaggle) is for local Flower/Ray simulation runs. Keep free space on D: — the distro was
    once corrupted by a full disk and had to be reinstalled. Full freeze:
    `requirements-lock-wsl.txt`. The venv lives on the Linux filesystem because `/mnt/d` is slow
    from WSL. The git-ignored symlink `.venv-wsl` in the repo points to it. Ubuntu 26.04 has no
    `python3.13` apt package, so Python comes from uv:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
    uv python install 3.13.18
    uv venv --seed --python 3.13.18 ~/venvs/fedsdn
    ln -sfn ~/venvs/fedsdn .venv-wsl        # run from the repo root
    ~/venvs/fedsdn/bin/pip install -r requirements.txt "ray==2.55.1"
    ~/venvs/fedsdn/bin/python scripts/smoke_flower.py   # 2 clients, 2 rounds, FedAvg
    ~/venvs/fedsdn/bin/pip freeze > requirements-lock-wsl.txt
    ```
- **Simulation entry point: `flwr.simulation.run_simulation`** (Message API: `ClientApp` /
  `ServerApp`, see `scripts/smoke_flower.py`). It is deprecated but works in the pinned
  `flwr==1.43.0`; the deprecation warning at startup is expected. **Do not migrate to `flwr run`.**
- After the first Kaggle install, save `pip freeze` as `requirements-lock-kaggle.txt`.
- `.gitattributes` forces LF line endings, so editing from Windows and from WSL produces no
  line-ending diffs.
- Paths come from env vars, never hard-coded:
  - `FEDSDN_DATA` (default `./data`; Kaggle: `/kaggle/input/<dataset-slug>`)
  - `FEDSDN_RESULTS` (default `./results`; Kaggle: `/kaggle/working/results`)
- `data/` and `results/` are git-ignored. Tests use tiny synthetic data only.

## Layout

```
src/fedsdn/data/      preprocess.py, partition.py, scaling.py
src/fedsdn/strategies/ custom FedNova + SCAFFOLD; wrappers for built-in Flower strategies
src/fedsdn/           models.py, client.py, metrics.py, checkpoint.py, run.py
configs/              base.yaml, tuning/, sweeps/
scripts/              build_dataset, make_partitions, expand_sweep, aggregate_results
kaggle/               driver notebook (install pins, set env vars, run config list)
tests/
```

## Raw data: InSDN

`data/raw/insdn/{Normal_data,OVS,metasploitable-2}.csv`, 84 columns, CICFlowMeter format.

- Column names and `Label` values have stray whitespace (`"DDoS "`) → strip both.
- `Timestamp` is **day-first**, mixed resolution, exactly two formats: most rows are 24-hour
  `D/M/YYYY H:MM` (`5/2/2020 13:58`, minute resolution); 85,811 rows are 12-hour with AM/PM
  `DD/MM/YYYY hh:MM:SS AM|PM` (`31/12/2019 02:25:57 PM`, second resolution). Parse explicitly
  with only these two formats; record resolution per row. No time zone in source → read as UTC.
- No NaN/inf at time of setup (still check in code).
- **Row-count check vs README** (record expected, observed and difference in `meta.json`):

  | File | README | Observed |
  |---|---|---|
  | Normal_data.csv | 68,424 | 68,424 |
  | OVS.csv | 136,743 | 138,722 |
  | metasploitable-2.csv | 138,772 | 136,743 |

  The README values for OVS and metasploitable-2 appear swapped, with a further 50-row difference.

## Preprocessing rules (Section III-C + project decisions) — apply in this order

1. Load all CSVs; add `row_id` = stable global index in (file name, file row) order; record raw counts.
2. Strip whitespace; parse `Timestamp` → `ts` (UTC) and `ts_resolution_s` (1 or 60).
3. Drop rows with missing or infinite values (count recorded).
4. **Identifiers are removed from the feature set**: `Flow ID`, `Src IP`, `Dst IP`, `Src Port`,
   `Timestamp`. **`Dst Port` is kept** as a numeric feature. `ts`, `src_ip`, `dst_ip` are retained
   as non-feature metadata only (for the split and the grouped-split fallback), never fed to the model.
5. **Deduplicate on feature columns + label** (after step 4):
   - First, every group whose features are identical but whose labels differ is dropped **entirely**
     (all rows of that group).
   - Then, among exact feature+label duplicates, keep the **earliest** flow (`ts`, ties → `row_id`).
   - Record in `meta.json`: rows dropped as conflicting, rows dropped as duplicates, and **class
     counts after dedup** (also print them).
6. Label map: `y_bin` (Normal=0, every attack=1) and `family` for the multi-class task.
7. **Rare classes**: families with < 500 flows after dedup are merged into one `Rare` class; if
   `Rare` still has < 500 flows, those flows get `y_multi = -1` (excluded from the multi-class task)
   but stay in the binary task as attack. At setup, before dedup, Web-Attack (192), BOTNET (164) and
   U2R (17) already qualify; after dedup **DDoS drops to 1,520** (still ≥ 500; the rule is applied
   to whatever falls below 500). InSDN result: Rare = BOTNET 164 + Web-Attack 183 + U2R 17 = 364
   → excluded (`y_multi = -1`).
8. **Class cap 20,000**: classes larger than the cap are thinned uniformly in time order (sort by
   `ts`, `row_id`; keep evenly spaced positions). Record pre- and post-cap counts.
9. **Split per family, between minutes** (non-capped rows; capped rows get `split = capped`):
   Floor every `ts` to its minute. A **minute cut** splits **between** two consecutive minutes
   (train = minutes ≤ cut), so no minute is on both sides. No flows are discarded (no gap).
   Per family, the first rule that succeeds is used (ties → earliest cut):
   - **(a) `minute_cut`**: cuts with test share in **[10%, 35%]** and **≥ 100 test flows**; take
     the one whose test share is closest to **20%**.
   - **(b) `minute_cut_relaxed`**: cuts with **≥ 100 test flows** and test share **≤ 50%**; take
     the one whose test share is closest to the range [10%, 35%].
   - **(c) `grouped_src_dst_pair`**: whole unordered src–dst address pairs go to test, with the
     constraints of (a) (share in [10%, 35%], ≥ 100 test flows, closest to 20%; exact subset-sum,
     ties → smaller test, pairs taken deterministically in name order). Record **pairs-per-flow**
     (distinct pairs / flows); **flag > 0.5** as degenerate grouping (e.g. spoofed sources).
   - **(d) `unsplit_all_train`**: the whole family goes to train (listed in `unsplit_families`).
   - **Rare families** (merged into `Rare` by step 7: BOTNET, Web-Attack, U2R) are split **each
     separately** by the same rules **without the 100-flow minimum**, except that in (b) the test
     share must also be **≥ 10%** (so (b) can only pick a share in (35%, 50%]).
   - Record per family: method, cut minutes or test pairs, realized test share, distinct minutes
     in train / test / shared, pairs-per-flow + flag (grouped only).
   - InSDN result (accepted): Normal, DoS (10.8%, only acceptable cut), Probe, U2R, Web-Attack →
     (a); DDoS → (b) at 42.7% (its sources are spoofed: one pair per flow, so (c) would be a
     random split); BFA → (c), pairs/flow 0.002; BOTNET → (d).
10. Drop feature columns that are constant over the training pool (12 at setup, e.g.
    `Fwd PSH Flags`, `Init Fwd Win Byts`); one-hot encode `Protocol`.
11. **No scaling in preprocessing.** Standardization happens inside each federation from per-client
    counts, sums and sums of squares, aggregated at the server (`data/scaling.py`).
12. The test set is never seen by any client. Each client holds out 10% of its shard for validation.
    Class imbalance → **class-weighted cross-entropy**, never oversampling.

## Data file format

`data/processed/<ds>/flows.parquet` — one row per flow surviving steps 1–5 (capped and excluded
rows are kept, marked, so counts are auditable):

| column | type | notes |
|---|---|---|
| `row_id` | int64 | stable join key for partitions |
| `ts` | timestamp[UTC] | metadata only |
| `ts_resolution_s` | int8 | 1 or 60 |
| `src_ip`, `dst_ip` | string | metadata only |
| `family` | string | original attack family (`Normal` for benign) |
| `y_multi` | int16 | class index, `-1` = excluded from multi-class |
| `y_bin` | int8 | 0 benign, 1 attack |
| `capped_out` | bool | removed by the 20k cap |
| `split` | string | `train` / `test` / `capped` (`capped` ⇔ `capped_out`) |
| `f_*` | float32 | features, unscaled, constant columns dropped, protocol one-hot |

`data/processed/<ds>/meta.json`: source file hashes, README vs observed row counts, NaN/inf drops,
conflict drops, duplicate drops, class counts after dedup / after rare rule / after cap / per split,
per-family split (method, cut / test pairs, test share, distinct minutes train / test / shared,
pairs-per-flow + degenerate flag for grouped families),
unsplit families, feature list, dropped columns + reason,
class index map, rare-class decision.

## Partition file format

`data/partitions/<ds>/<scheme>_<param>_K<K>_s<seed>.parquet`, e.g. `P1_a0.5_K8_s3`, `P3_m1_K8_s0`,
`P4_a0.5b0.5_K8_s0`. Training-pool rows only.

| column | type |
|---|---|
| `row_id` | int64 |
| `client_id` | int16 (0..K−1) |
| `is_val` | bool (10% local validation) |

Sidecar `.json`: scheme + parameters, K, seed, `flows_sha256` (hash of `flows.parquet`; runs refuse
to start on mismatch), attempts used (≤100), whether rows were reassigned to meet `n_min = 100`,
per-client class counts, mean pairwise JSD of label distributions, CV of shard sizes, (P3) family →
client assignment. Partitions are built on `family` labels and reused unchanged for the binary task.
Rows with `y_multi = -1` (the excluded Rare group) are treated as **one family of their own** during
partitioning, so the same shards serve both tasks. In the multi-class task those rows stay in the
shard and are masked out (see Training rules).

Schemes (`data/partition.py`; "dealing" = shuffle, then consecutive chunks sized by
largest-remainder rounding of the proportions; RNG stream per (setting tag, seed)):
- **P0** shuffle, equal shards. **P1** per family `q_c ~ Dir_K(α)`, α ∈ {0.1, 0.5, 1, 10}, family
  dealt by `q_c`. **P2** `s ~ Dir_K(β)`, β ∈ {0.5, 1}, all rows dealt by `s`.
- **P3** benign dealt equally over all K; attack families in a seeded order, round-robin,
  `m_eff = max(m, ⌈F/K⌉)` distinct families per client (m ∈ {1, 2}); a family is dealt equally
  among its holders.
- **P4** (approved method, not specified in the paper): `q_c ~ Dir_K(0.5)` per family, one
  `s ~ Dir_K(0.5)`; family c dealt by `q̃_c,k = q_c,k·s_k / Σ_j q_c,j·s_j`. Uniform `s` → P1,
  uniform `q` → P2 (in expectation).
- `n_min = 100` applies to the whole shard (validation included). After 100 failed draws, the
  draw with the largest minimum shard is kept and random rows move from the current largest shard.
- Validation: per client `round(0.1·n_k)` rows, allocated over families proportionally.
- JSD: base 2, over per-client family distributions of the whole shard; CV uses population std.
- Scaling (`data/scaling.py`): float64 count / sum / sum of squares on training rows (validation
  excluded); population std; a feature with `var ≤ 1e-12·E[x²]` is constant → std 1.

## Run config (YAML)

```yaml
run_id: null            # null → first 12 hex chars of sha1 of the canonical config (sorted keys, minus run_id)
dataset: insdn
task: multi             # multi | binary
partition: {scheme: P1, alpha: 0.5, beta: null, m: null}
K: 8
E: 1
R: 100
batch_size: 256
model: mlp              # mlp | cnn1d
seed: 0
strategy:
  name: fedavg          # fedavg|fedprox|scaffold|fednova|fedadam|krum|median|trimmed_mean
  lr: 0.05              # client SGD rate
  mu: null              # FedProx
  server_lr: null       # FedAdam
  eps: null             # FedAdam
  f_hat: null           # robust rules; null → floor(K/4) (Krum uses 0 at K=4)
attack: {f: 0, type: null}   # type: label_flip | sign_flip | targeted
checkpoint_every: 1
```

## Results schema

`$FEDSDN_RESULTS/<run_id>/`:

- `config.yaml` — resolved config as run.
- `rounds.csv` — one row per round: `run_id, round, macro_f1, accuracy, precision, recall, fpr,
  rare_recall, worst_client_f1, client_f1_std, n_present_classes_min, bytes_up, bytes_down,
  client_cpu_s, server_s, wall_s`
- `per_class.csv` — `run_id, round, class, recall, support`
- `summary.csv` — one row: config fields flattened + final metrics, `t95`, `t95_censored`,
  `auc_f1_rounds`
- `ckpt.pt` — checkpoint (below). `status` — `RUNNING` | `DONE`.

## Training rules

- Model: MLP d→128→64→C, ReLU, dropout 0.2 (logits + `CrossEntropyLoss`); 1D-CNN secondary.
- **Plain mini-batch SGD, momentum = 0, no weight decay**, so local steps τ_k = E·⌈n_k/B⌉ are exact
  (FedNova needs this). Batch 256.
- **Class-weighted cross-entropy** loss; no oversampling. Weights use **global** training counts:
  each client reports its training-row class counts (validation rows excluded), the server sums
  them, and `weight_c = n_train / (C · n_c)`, where `C` is the number of classes in the task.
  Computed once per run, before round 1, and the same weights go to every client. The binary task
  works the same way with `C = 2`. In the multi-class task, `n_train` and `n_c` count only rows with
  `y_multi ≥ 0`.
- **Rare rows (`y_multi = -1`)**: in the multi-class task they are masked out of the loss and out of
  every metric. In the binary task they are ordinary attack rows.
- FedProx term is added in client code; SCAFFOLD client control variates persist outside the
  ephemeral Flower client, keyed by `client_id`.
- Unit tests: FedNova == FedAvg when all τ_k equal; SCAFFOLD == FedAvg with control variates frozen
  at zero.

## Resumability (every run, keyed by run_id)

- `ckpt.pt` after every `checkpoint_every` rounds: completed round, global weights, server optimizer
  state (FedAdam m, v), SCAFFOLD server + per-client control variates, malicious-client set, RNG
  states (python, numpy, torch).
- On start: if `status == DONE` → skip. If a checkpoint exists → load it, truncate `rounds.csv` /
  `per_class.csv` rows with `round > ckpt.round`, continue. Otherwise start fresh.
- Writes are atomic (write temp file, then `os.replace`).
- Test: a run interrupted and resumed must give results identical to an uninterrupted run.
- Kaggle sessions are time-limited: drivers take a list of configs and rely on this to continue.

## Reduced study (time-boxed; plain PyTorch, no Flower/Ray)

`src/fedsdn/sim.py` + `scripts/run_reduced.py` (runs, resumable via per-run JSON markers,
`results/progress.log`) + `scripts/report_reduced.py` (`results/report/`). InSDN, multi-class only,
K = 8, E = 1, R = 30, batch 256, SGD lr 0.05, MLP; partitions P0, P1 α=0.5, P3 m=1, P4; seeds 0–2.
Strategies: FedAvg, FedProx (μ 0.01), FedNova, FedAdam (server lr 0.01, β 0.9/0.99, ε 1e-3),
Krum / Median / Trimmed Mean (f̂ = 2, clients weighted equally). Poisoning (chrono only): P4, f = 2,
sign flip (−update) and targeted (malicious clients relabel attack rows as Normal).
- **Protocols**: `chrono` (primary; `processed/insdn`, `partitions/insdn`) and `random` (secondary
  random stratified 80/20 split of the same rows, by partition family, seed 0;
  `processed/insdn_random`, `partitions/insdn_random`; `scripts/make_random_split.py`). The random
  protocol is a leakage diagnostic only, never compared directly with chrono.
- **Headline metric = val-selected test macro-F1**: every round logs global validation macro-F1
  (pooled over all clients' validation sets); the run reports test metrics at the best-validation
  round (ties → earliest). The test set is never used for selection. Also reported: final round
  and AUC (mean test macro-F1 over rounds). Centralized baseline holds out a stratified 10% of the
  training pool as validation.
- **Rounds per protocol**: chrono and poisoning at R = 30 (`results/chrono/`). The random protocol
  headline is R = 100 for all 7 strategies with a 100-epoch centralized baseline
  (`run_reduced.py --suite r100` → `results/r100/random/`); the R = 30 random runs
  (`results/random/`) are kept only for the R = 30 vs R = 100 comparison (R = 30 was too short:
  val-selected rounds sat near the end and T95 was censored almost everywhere).
- T95 = first round with test macro-F1 ≥ 0.95 × the protocol's mean centralized val-selected
  macro-F1, centralized trained for as many epochs as the protocol has rounds (censored
  otherwise). Bytes per round = params × 4 B × 2K.
- Diagnostics (`scripts/diag_central.py`, chrono, seed 0): the baseline is neither underfitting
  (train F1 0.88) nor undertrained (100 epochs: test F1 0.52); random split gives 0.95. The gap is
  distribution shift: BFA test recall 0 (held-out address pair), Probe ≈ 0.3.
- Old chrono results without validation logging: `results/archive_chrono_v1_noval/`.
- Paper outputs: `scripts/paper_outputs.py` → `results/paper_tables/*.csv` (3 decimals, mean/std in
  separate columns; Friedman + Nemenyi on random R = 100 val-selected macro-F1, blocks =
  partition × seed) and `results/paper_figures/*.png` (300 dpi, 3.5 in single column, Okabe-Ito
  colors fixed per strategy, no titles). Never built from `results/dry`.
  Font: STIXGeneral (Times-like, bundled with matplotlib). Architecture / pipeline diagrams:
  `scripts/paper_diagrams.py` → fig0_system_architecture, fig_model_architecture, fig_pipeline.
- Flower validation (`scripts/flower_validation.py`, WSL venv, `PYTHONPATH=src`): random protocol,
  P4, seed 0, R = 30; built-in Flower FedAvg / FedMedian and a custom Flower strategy calling our
  FedAdam; clients reuse `sim.local_train`; scaling + class weights via a federated QUERY round.
  Matches the plain loop run with `RunConfig(per_step_seed=True)` (dropout re-seeded per
  (round, client), as in Flower's per-actor processes) to max |ΔF1| ≤ 0.0012; serialized
  messages are 1.028 × the derived params × 4 B × 2K. Outputs: `paper_tables/flower_validation.csv`,
  `paper_figures/fig7_flower_vs_loop.png`, `results/flower_validation/`.
