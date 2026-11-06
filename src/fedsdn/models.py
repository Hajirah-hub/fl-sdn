"""Detection model: MLP d -> 128 -> 64 -> C, ReLU, dropout 0.2, logits out."""

from torch import nn


class MLP(nn.Module):
    def __init__(self, d: int, C: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, C),
        )

    def forward(self, x):
        return self.net(x)
