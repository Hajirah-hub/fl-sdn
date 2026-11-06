"""Smoke test: Flower simulation on Ray, 2 clients, 2 rounds, FedAvg, linear model, random data.

Run (WSL or Kaggle):  python scripts/smoke_flower.py
Checks only that flwr + ray + torch work together; it has nothing to do with the study itself.
"""

import torch
from torch import nn

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.simulation import run_simulation

N_CLIENTS, N_ROUNDS, D, C, N = 2, 2, 10, 2, 200


def make_model() -> nn.Module:
    return nn.Linear(D, C)


def client_data(pid: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(pid)
    x = torch.randn(N, D, generator=g)
    y = torch.randint(0, C, (N,), generator=g)
    return x, y


client_app = ClientApp()


@client_app.train()
def train(msg: Message, context: Context) -> Message:
    pid = int(context.node_config["partition-id"])
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    x, y = client_data(pid)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for xb, yb in zip(x.split(50), y.split(50)):
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss.item(), "num-examples": N}),
    })
    return Message(content=content, reply_to=msg)


@client_app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    pid = int(context.node_config["partition-id"])
    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    x, y = client_data(pid)
    with torch.no_grad():
        logits = model(x)
        loss = float(nn.functional.cross_entropy(logits, y))
        acc = float((logits.argmax(1) == y).float().mean())
    content = RecordDict({"metrics": MetricRecord({"eval_loss": loss, "eval_acc": acc, "num-examples": N})})
    return Message(content=content, reply_to=msg)


server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    torch.manual_seed(0)
    strategy = FedAvg(min_train_nodes=N_CLIENTS, min_evaluate_nodes=N_CLIENTS,
                      min_available_nodes=N_CLIENTS)
    result = strategy.start(grid=grid, initial_arrays=ArrayRecord(make_model().state_dict()),
                            num_rounds=N_ROUNDS)
    print("SMOKE RESULT:", result)


if __name__ == "__main__":
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=N_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
