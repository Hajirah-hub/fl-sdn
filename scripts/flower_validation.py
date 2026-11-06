"""Validation: the plain PyTorch loop (fedsdn.sim) vs a real Flower simulation (Ray backend).

    PYTHONPATH=src ~/venvs/fedsdn/bin/python scripts/flower_validation.py      # WSL venv (Ray)

Random protocol, P4, seed 0, K = 8, E = 1, R = 30. Three configurations:
  fedavg  - Flower's built-in FedAvg (cross-checks fedsdn.strategies.aggregate.fedavg)
  fedadam - custom Flower strategy calling fedsdn.strategies.aggregate.FedAdam
  median  - Flower's built-in FedMedian (median of client weights = global + median of updates)
Each ClientApp is one controller: it loads its own shard by partition-id = client_id and trains
with fedsdn.sim.local_train (same MLP, local SGD, class weights, data order). Scaling and class
weights come from a federated QUERY round (client count / sum / sum of squares + class counts).

Compared against (i) the plain loop re-run with per-step seeding (dropout re-seeded per
(round, client), as in Flower, where each client trains in its own process) and (ii) the original
plain-loop R = 30 random-protocol runs (one global RNG stream). Message sizes are the serialized
protobuf sizes of every train message sent and received.

Writes results/paper_tables/flower_validation.csv, results/flower_validation/*.csv|json and
results/paper_figures/fig7_flower_vs_loop.png.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MessageType, MetricRecord, RecordDict  # noqa: E402
from flwr.clientapp import ClientApp  # noqa: E402
from flwr.common.serde import message_to_proto  # noqa: E402
from flwr.serverapp import Grid, ServerApp  # noqa: E402
from flwr.serverapp.strategy import FedAvg, FedMedian  # noqa: E402
from flwr.simulation import run_simulation  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402

from fedsdn.data.scaling import FeatureStats, aggregate_feature_stats, client_feature_stats  # noqa: E402
from fedsdn.models import MLP  # noqa: E402
from fedsdn.sim import RunConfig, class_names, class_weights, load_flows, local_train, macro_f1, predict, run, step_seed  # noqa: E402
from fedsdn.strategies import aggregate as agg  # noqa: E402

DATA = Path(os.environ.get("FEDSDN_DATA", ROOT / "data"))
RES = Path(os.environ.get("FEDSDN_RESULTS", ROOT / "results"))
PROCESSED = DATA / "processed" / "insdn_random"
PARTITIONS = DATA / "partitions" / "insdn_random"
PARTITION, SEED, K, R = "P4_a0.5b0.5", 0, 8, 30
STRATEGIES = ("fedavg", "fedadam", "median")
IMPL = {"fedavg": "Flower built-in FedAvg", "fedadam": "custom Flower strategy -> fedsdn FedAdam",
        "median": "Flower built-in FedMedian"}
OUT = RES / "flower_validation"


# --------------------------------------------------------------------------------------------
# client side (runs in Ray actors)
# --------------------------------------------------------------------------------------------

_CACHE: dict = {}


def client_shard(cfg: ConfigRecord, k: int) -> dict:
    """This controller's training rows (validation excluded), by partition-id = client_id."""
    key = (cfg["processed"], cfg["partition_file"], k)
    if key not in _CACHE:
        torch.set_num_threads(1)
        info, X_raw, _ = load_flows(cfg["processed"])
        part = pd.read_parquet(cfg["partition_file"])
        pos = pd.Series(np.arange(len(info)), index=info["row_id"].to_numpy())
        rows = pos.loc[part["row_id"].to_numpy()].to_numpy()
        cid, is_val = part["client_id"].to_numpy(), part["is_val"].to_numpy()
        tr_all = rows[(cid == k) & ~is_val]
        y_all = info["y_multi"].to_numpy()
        tr = tr_all[y_all[tr_all] >= 0]
        _CACHE[key] = {"X_all_raw": X_raw[tr_all], "X_raw": X_raw[tr], "y": y_all[tr]}
    return _CACHE[key]


client_app = ClientApp()


@client_app.query()
def query(msg: Message, context: Context) -> Message:
    k = int(context.node_config["partition-id"])
    cfg = msg.content["config"]
    sh = client_shard(cfg, k)
    st = client_feature_stats(sh["X_all_raw"])                 # validation excluded, y = -1 kept
    counts = np.bincount(sh["y"], minlength=int(cfg["C"]))
    rec = MetricRecord({"count": st.count, "sum": st.sum.tolist(), "sumsq": st.sumsq.tolist(),
                        "class_counts": [int(c) for c in counts]})
    return Message(content=RecordDict({"stats": rec}), reply_to=msg)


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    k = int(context.node_config["partition-id"])
    cfg = msg.content["config"]
    sh = client_shard(cfg, k)
    r = int(cfg["server-round"])
    mean, std = np.array(cfg["mean"]), np.array(cfg["std"])
    X = torch.from_numpy(((sh["X_raw"] - mean) / std).astype(np.float32))
    y = torch.from_numpy(sh["y"].astype(np.int64))
    cw = torch.tensor(cfg["cw"], dtype=torch.float32)
    model = MLP(X.shape[1], int(cfg["C"]))
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    w0 = parameters_to_vector(model.parameters()).detach().clone()
    rc = RunConfig(kind="fed", seed=int(cfg["seed"]), lr=float(cfg["lr"]), batch=int(cfg["batch"]), E=int(cfg["E"]))
    ss = step_seed(rc.seed, r, k)
    t0 = time.perf_counter()
    delta, steps = local_train(model, w0, X, y, cw, rc, torch.Generator().manual_seed(ss), prox=False, torch_seed=ss)
    vector_to_parameters(w0 + delta, model.parameters())
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()),
                          "metrics": MetricRecord({"num-examples": len(y), "steps": steps,
                                                   "train_s": time.perf_counter() - t0})})
    return Message(content=content, reply_to=msg)


# --------------------------------------------------------------------------------------------
# server side
# --------------------------------------------------------------------------------------------


def msg_bytes(m: Message) -> int:
    return len(message_to_proto(m).SerializeToString())


class Measured:
    """Mixin: record serialized sizes of train messages sent (download) and received (upload)."""

    def configure_train(self, server_round, arrays, config, grid):
        msgs = list(super().configure_train(server_round, arrays, config, grid))
        self.log.setdefault(server_round, {})["down_bytes"] = sum(msg_bytes(m) for m in msgs)
        self.current = arrays
        return msgs

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        self.log.setdefault(server_round, {})["up_bytes"] = sum(msg_bytes(m) for m in replies)
        self.log[server_round]["n_replies"] = len(replies)
        return self._aggregate(server_round, replies)

    def _aggregate(self, server_round, replies):
        return super().aggregate_train(server_round, replies)


class MeasuredFedAvg(Measured, FedAvg):
    pass


class MeasuredFedMedian(Measured, FedMedian):
    pass


def to_vec(ar: ArrayRecord) -> torch.Tensor:
    return torch.cat([torch.from_numpy(np.asarray(a)).reshape(-1) for a in ar.to_numpy_ndarrays()])


class OurFedAdam(Measured, FedAvg):
    """Custom Flower strategy whose aggregation is fedsdn.strategies.aggregate.FedAdam."""

    def __init__(self, template: torch.nn.Module, **kw):
        super().__init__(**kw)
        self.template = template
        self.opt = agg.FedAdam(0.01, 0.9, 0.99, 1e-3)

    def _aggregate(self, server_round, replies):
        valid, _ = self._check_and_log_replies(replies, is_train=True)
        replies = sorted(valid, key=lambda m: m.metadata.src_node_id)
        w = to_vec(self.current)
        U = torch.stack([to_vec(m.content["arrays"]) - w for m in replies])
        n = torch.tensor([float(m.content["metrics"]["num-examples"]) for m in replies], dtype=torch.float64)
        vector_to_parameters(w + self.opt.step(U, n), self.template.parameters())
        return ArrayRecord(self.template.state_dict()), None


server_app = ServerApp()
RESULT: dict = {}


@server_app.main()
def main_server(grid: Grid, context: Context) -> None:
    name = os.environ["FEDSDN_FLWR_STRATEGY"]
    meta = json.loads((PROCESSED / "meta.json").read_text())
    C = len(class_names(meta))
    info, X_raw, _ = load_flows(str(PROCESSED))
    y_all = info["y_multi"].to_numpy()
    test = np.flatnonzero((info["split"] == "test").to_numpy() & (y_all >= 0))

    data_cfg = {"processed": str(PROCESSED), "partition_file": str(PARTITIONS / f"{PARTITION}_K{K}_s{SEED}.parquet"),
                "C": C, "seed": SEED, "lr": 0.05, "batch": 256, "E": 1}
    # federated statistics round (QUERY): global standardization + global class weights
    nodes = sorted(grid.get_node_ids())
    while len(nodes) < K:
        time.sleep(0.5)
        nodes = sorted(grid.get_node_ids())
    q = [Message(content=RecordDict({"config": ConfigRecord(data_cfg)}), dst_node_id=n,
                 message_type=MessageType.QUERY) for n in nodes]
    replies = list(grid.send_and_receive(q))
    stats = [FeatureStats(int(m.content["stats"]["count"]), np.array(m.content["stats"]["sum"]),
                          np.array(m.content["stats"]["sumsq"])) for m in replies]
    mean, std = aggregate_feature_stats(stats)
    counts = np.sum([np.array(m.content["stats"]["class_counts"]) for m in replies], axis=0)
    cw = class_weights(counts)
    X_test = torch.from_numpy(((X_raw[test] - mean) / std).astype(np.float32))
    y_test = y_all[test]

    torch.manual_seed(SEED)                                   # same initial model as the plain loop
    model = MLP(X_raw.shape[1], C)
    common = dict(fraction_evaluate=0.0, min_train_nodes=K, min_available_nodes=K, min_evaluate_nodes=0)
    if name == "fedavg":
        strat = MeasuredFedAvg(**common)
    elif name == "median":
        strat = MeasuredFedMedian(**common)
    else:
        strat = OurFedAdam(MLP(X_raw.shape[1], C), **common)
    strat.log = {}
    evals = {}
    marks = {"last_end": time.perf_counter()}

    def evaluate_fn(server_round: int, arrays: ArrayRecord):
        t0 = time.perf_counter()
        model.load_state_dict(arrays.to_torch_state_dict())
        f1 = macro_f1(y_test, predict(model, X_test), list(range(C)))
        t1 = time.perf_counter()
        evals[server_round] = {"macro_f1": f1, "round_wall_s": t0 - marks["last_end"], "eval_s": t1 - t0}
        marks["last_end"] = t1
        return MetricRecord({"test_macro_f1": f1})

    train_cfg = ConfigRecord({**data_cfg, "mean": mean.tolist(), "std": std.tolist(), "cw": cw.tolist()})
    marks["last_end"] = time.perf_counter()
    strat.start(grid=grid, initial_arrays=ArrayRecord(model.state_dict()), num_rounds=R,
                train_config=train_cfg, evaluate_fn=evaluate_fn)
    RESULT.update(strategy=name, evals=evals, log=strat.log, n_params=int(sum(p.numel() for p in model.parameters())))


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


def run_flower(name: str) -> pd.DataFrame:
    os.environ["FEDSDN_FLWR_STRATEGY"] = name
    RESULT.clear()
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=K,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}})
    rows = []
    for r in range(1, R + 1):
        e, lg = RESULT["evals"][r], RESULT["log"].get(r, {})
        rows.append({"round": r, "macro_f1": e["macro_f1"], "round_wall_s": e["round_wall_s"],
                     "up_bytes": lg.get("up_bytes"), "down_bytes": lg.get("down_bytes"),
                     "n_replies": lg.get("n_replies")})
    df = pd.DataFrame(rows)
    df.attrs["n_params"] = RESULT["n_params"]
    return df


def run_plain(name: str) -> pd.DataFrame:
    torch.set_num_threads(1)
    meta = json.loads((PROCESSED / "meta.json").read_text())
    cfg = RunConfig(kind="fed", strategy=name, partition=PARTITION, seed=SEED, protocol="random", R=R,
                    per_step_seed=True)
    _, rounds = run(cfg, str(PROCESSED), str(PARTITIONS), meta)
    return rounds[["round", "macro_f1", "wall_s"]]


def summarize(name: str, n_params: int) -> dict:
    """One table row from results/flower_validation/<name>_rounds.csv."""
    cmp_ = pd.read_csv(OUT / f"{name}_rounds.csv")
    d_m = (cmp_.flower_macro_f1 - cmp_.plain_matched_macro_f1).abs()
    d_o = (cmp_.flower_macro_f1 - cmp_.plain_original_macro_f1).abs()
    d_mo = (cmp_.plain_matched_macro_f1 - cmp_.plain_original_macro_f1).abs()
    derived = n_params * 4 * 2 * K
    up, down = cmp_.flower_up_bytes.mean(), cmp_.flower_down_bytes.mean()
    last = cmp_.iloc[-1]
    return {
        "strategy": name, "flower_implementation": IMPL[name], "protocol": "random", "partition": PARTITION,
        "seed": SEED, "K": K, "R": R, "E": 1,
        "final_f1_flower": last.flower_macro_f1,
        "final_f1_plain_matched": last.plain_matched_macro_f1,
        "final_f1_plain_original": last.plain_original_macro_f1,
        "final_diff_flower_vs_matched": last.flower_macro_f1 - last.plain_matched_macro_f1,
        "max_abs_diff_flower_vs_matched": d_m.max(), "mean_abs_diff_flower_vs_matched": d_m.mean(),
        "final_diff_flower_vs_original": last.flower_macro_f1 - last.plain_original_macro_f1,
        "max_abs_diff_flower_vs_original": d_o.max(), "mean_abs_diff_flower_vs_original": d_o.mean(),
        "max_abs_diff_matched_vs_original": d_mo.max(),
        "n_params": n_params, "derived_bytes_per_round": derived,
        "flower_upload_bytes_per_round": up, "flower_download_bytes_per_round": down,
        "flower_total_bytes_per_round": up + down, "flower_over_derived_ratio": (up + down) / derived,
        "flower_wall_s_per_round": cmp_.flower_round_wall_s.mean(),
        "plain_wall_s_per_round": cmp_.plain_round_wall_s.mean(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in STRATEGIES:
        print(f"=== {name}: plain loop (per-step seed)", flush=True)
        plain = run_plain(name)
        print(f"=== {name}: Flower simulation", flush=True)
        fl = run_flower(name)
        orig = pd.read_csv(RES / "random" / "runs" / f"random_{name}_{PARTITION}_s{SEED}_rounds.csv")
        cmp_ = pd.DataFrame({"round": fl["round"], "flower_macro_f1": fl["macro_f1"],
                             "plain_matched_macro_f1": plain["macro_f1"].to_numpy(),
                             "plain_original_macro_f1": orig["macro_f1"].to_numpy(),
                             "flower_round_wall_s": fl["round_wall_s"], "plain_round_wall_s": plain["wall_s"].to_numpy(),
                             "flower_up_bytes": fl["up_bytes"], "flower_down_bytes": fl["down_bytes"]})
        cmp_.to_csv(OUT / f"{name}_rounds.csv", index=False)
        rows.append(summarize(name, fl.attrs["n_params"]))
        print(pd.Series(rows[-1]).to_string(), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "flower_validation_full_precision.csv", index=False)
    write_table(table)
    plot()


def write_table(table: pd.DataFrame) -> None:
    """3 decimals, except F1 differences (6 decimals; they are far below 0.001)."""
    table = table.copy()
    diff_cols = [c for c in table.columns if "diff" in c]
    num = [c for c in table.select_dtypes("number").columns if c not in diff_cols]
    table[num] = table[num].round(3)
    table[diff_cols] = table[diff_cols].round(6)
    (RES / "paper_tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(RES / "paper_tables" / "flower_validation.csv", index=False)
    print(table.T.to_string())


def plot() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paper_outputs import COLOR, LABEL, SINGLE, save, style
    style()
    fig, axes = plt.subplots(3, 1, figsize=(SINGLE, 4.6), sharex=True)
    for ax, name in zip(axes, STRATEGIES):
        d = pd.read_csv(OUT / f"{name}_rounds.csv")
        ax.plot(d["round"], d.plain_original_macro_f1, color="#999999", lw=1.0, ls=":",
                label="Plain loop (original runs)")
        ax.plot(d["round"], d.flower_macro_f1, color=COLOR[name], lw=2.2, alpha=0.75, label="Flower (Ray)")
        ax.plot(d["round"], d.plain_matched_macro_f1, color="k", lw=0.9, ls="--",
                label="Plain loop (per-step seed)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test macro-F1")
        ax.text(0.98, 0.06, LABEL[name if name != "median" else "median"], transform=ax.transAxes,
                ha="right", va="bottom")
    axes[-1].set_xlabel("Round")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.42), ncol=2, columnspacing=0.9, handlelength=1.8)
    fig.tight_layout(h_pad=0.4)
    (RES / "paper_figures").mkdir(parents=True, exist_ok=True)
    save(fig, RES / "paper_figures" / "fig7_flower_vs_loop.png")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:                      # rebuild table + figure from saved rounds
        table = pd.DataFrame([summarize(n, 17_413) for n in STRATEGIES])
        table.to_csv(OUT / "flower_validation_full_precision.csv", index=False)
        write_table(table)
        plot()
    else:
        main()
