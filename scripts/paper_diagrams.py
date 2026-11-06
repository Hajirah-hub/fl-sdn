"""Architecture diagrams for the paper (300 dpi PNG, same fonts/colors as paper_outputs.py).

    python scripts/paper_diagrams.py

-> results/paper_figures/fig0_system_architecture.png  (double column)
   results/paper_figures/fig_model_architecture.png    (single column)
   results/paper_figures/fig_pipeline.png              (double column)
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_outputs import COLOR, DOUBLE, LABEL, SINGLE, STRATS, save, style  # noqa: E402

INK = "#222222"
FILL = "#F2F2F2"
FILL2 = "#E3ECF5"
MAL = "#D55E00"          # Okabe-Ito vermillion (colorblind-safe red)
TEXT_STRAT = {**COLOR, "trimmed_mean": "#9C8E00"}   # darker yellow so text stays readable


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc=FILL, ec=INK, lw=0.8, ls="-", r=0.05, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.9, ls="-", style_="-|>", ms=7, z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style_, mutation_scale=ms, color=color,
                                 lw=lw, ls=ls, zorder=z, shrinkA=0, shrinkB=0))


def text(ax, x, y, s, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    kw.setdefault("fontsize", 8)
    kw.setdefault("color", INK)
    ax.text(x, y, s, **kw)


# --------------------------------------------------------------------------------------------
# fig0: system architecture
# --------------------------------------------------------------------------------------------


def fig_system(out: Path) -> None:
    W, H = DOUBLE, 4.0
    fig, ax = canvas(W, H)
    K, x0, gap = 8, 0.1, 0.07
    cw = (W - 2 * x0 - (K - 1) * gap) / K
    malicious = {2, 5}                                   # C3 and C6, illustrative

    # aggregation server
    sx, sw, sy, sh = 1.2, 3.55, 3.0, 0.92
    box(ax, sx, sy, sw, sh, fc=FILL2, lw=1.0)
    text(ax, sx + sw / 2, sy + sh - 0.15, "Aggregation server", fontweight="bold")
    text(ax, sx + sw / 2, sy + sh - 0.33, "one strategy per run:")
    for ri, row in enumerate((STRATS[:4], STRATS[4:])):
        for i, st in enumerate(row):
            text(ax, sx + sw * (i + 0.5) / len(row), sy + 0.36 - ri * 0.2, LABEL[st],
                 color=TEXT_STRAT[st], fontweight="bold")

    # evaluation box fed by the global model
    ex, ew, ey, eh = 5.45, 1.6, 3.12, 0.68
    box(ax, ex, ey, ew, eh, fc="white", lw=0.8)
    text(ax, ex + ew / 2, ey + eh / 2, "Evaluation on\nheld-out test set\n(never seen by clients)",
         linespacing=1.1)
    arrow(ax, sx + sw, ey + eh / 2, ex, ey + eh / 2)
    text(ax, (sx + sw + ex) / 2, ey + eh / 2 - 0.17, "global\nmodel", linespacing=1.0)

    # malicious legend (right, below the evaluation box)
    ax.add_patch(Rectangle((5.5, 2.74), 0.2, 0.12, fc="white", ec=MAL, lw=1.4, ls="--"))
    text(ax, 5.76, 2.8, "malicious (poisoning study)", ha="left", color=MAL)

    # controllers, domains, arrows
    cy, ch = 1.2, 0.98
    dy, dh = 0.32, 0.6
    bus_up, bus_down = 2.62, 2.42
    for k in range(K):
        x = x0 + k * (cw + gap)
        mal = k in malicious
        box(ax, x, cy, cw, ch, fc="white", ec=MAL if mal else INK, lw=1.6 if mal else 0.8,
            ls="--" if mal else "-")
        text(ax, x + cw / 2, cy + ch - 0.13, f"C{k + 1}", fontweight="bold")
        text(ax, x + cw / 2, cy + ch - 0.36, "local flow data")
        ax.plot([x + 0.08, x + cw - 0.08], [cy + ch - 0.5, cy + ch - 0.5], color="#BBBBBB", lw=0.5)
        text(ax, x + cw / 2, cy + 0.24, "local training\n(E epochs, SGD)", linespacing=1.05)
        # data-plane domain: three switches in a triangle
        box(ax, x, dy, cw, dh, fc=FILL, ec="#888888", lw=0.6, r=0.04)
        pts = [(x + cw * 0.27, dy + 0.4), (x + cw * 0.73, dy + 0.4), (x + cw * 0.5, dy + 0.15)]
        for (xa, ya), (xb, yb) in ((pts[0], pts[1]), (pts[0], pts[2]), (pts[1], pts[2])):
            ax.plot([xa, xb], [ya, yb], color="#555555", lw=0.5, zorder=2)
        for xs, ys in pts:
            ax.add_patch(Rectangle((xs - 0.09, ys - 0.04), 0.18, 0.08, fc="white", ec="#555555", lw=0.5, zorder=3))
        arrow(ax, x + cw / 2, dy + dh, x + cw / 2, cy, color="#555555", lw=0.7, ms=5)
        arrow(ax, x + cw * 0.33, cy + ch, x + cw * 0.33, bus_up, color=INK, lw=0.9)
        arrow(ax, x + cw * 0.67, bus_down, x + cw * 0.67, cy + ch, color="#777777", lw=0.9, ls="--")
    xl, xr = x0 + cw * 0.33, x0 + (K - 1) * (cw + gap) + cw * 0.33
    ax.plot([xl, xr], [bus_up, bus_up], color=INK, lw=0.9)
    arrow(ax, sx + sw * 0.42, bus_up, sx + sw * 0.42, sy, color=INK, lw=1.0)
    xl2, xr2 = x0 + cw * 0.67, x0 + (K - 1) * (cw + gap) + cw * 0.67
    ax.plot([xl2, xr2], [bus_down, bus_down], color="#777777", lw=0.9, ls="--")
    ax.plot([sx + sw * 0.58, sx + sw * 0.58], [sy, bus_down], color="#777777", lw=0.9, ls="--")
    text(ax, sx + sw * 0.42 - 0.08, 2.8, r"model update $\Delta w_k$", ha="right")
    text(ax, sx + sw * 0.58 + 0.08, 2.8, r"global model $w^{(t+1)}$", ha="left", color="#555555")
    text(ax, W / 2, 0.13, "Data plane: one switch domain per controller  —  raw traffic never leaves a controller",
         style="italic")
    save(fig, out)


# --------------------------------------------------------------------------------------------
# model architecture
# --------------------------------------------------------------------------------------------


def fig_model(out: Path) -> None:
    W, H = SINGLE, 1.95
    fig, ax = canvas(W, H)
    layers = [
        ("Input", "68 features", "–", "white"),
        ("Dense 128", "ReLU\ndropout 0.2", "8,832", FILL2),
        ("Dense 64", "ReLU\ndropout 0.2", "8,256", FILL2),
        ("Dense 5", "softmax\n5 classes", "325", FILL),
    ]
    n = len(layers)
    bw, gap, y, bh = 0.66, 0.2, 0.62, 0.95
    x0 = (W - (n * bw + (n - 1) * gap)) / 2
    for i, (name, sub, params, fc) in enumerate(layers):
        x = x0 + i * (bw + gap)
        box(ax, x, y, bw, bh, fc=fc, lw=0.8)
        text(ax, x + bw / 2, y + bh - 0.18, name, fontweight="bold")
        text(ax, x + bw / 2, y + 0.33, sub, linespacing=1.05)
        text(ax, x + bw / 2, y - 0.14, params)
        if i < n - 1:
            arrow(ax, x + bw, y + bh / 2, x + bw + gap, y + bh / 2, ms=6)
    text(ax, x0 - 0.04, y - 0.14, "params:", ha="right")
    text(ax, W / 2, 0.2, "Total 17,413 parameters (68 KB in float32)", fontweight="bold")
    text(ax, W / 2, H - 0.12, "classes: Normal, BFA, DDoS, DoS, Probe", fontsize=8)
    save(fig, out)


# --------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------


def fig_pipeline(out: Path) -> None:
    W, H = DOUBLE, 2.75
    fig, ax = canvas(W, H)
    bw, bh, gx = 2.12, 0.92, 0.26
    x0 = (W - (3 * bw + 2 * gx)) / 2
    top, bot = 1.72, 0.42
    stages = [
        (0, top, "1. Dataset", "InSDN only\n3 CSVs, 343,889 flows\n84 CICFlowMeter columns", "white"),
        (1, top, "2. Preprocessing", "drop identifiers; dedup (feature + label,\nconflicts removed); rare-family rule;\n"
                                     "20k class cap; per-family minute split", FILL),
        (2, top, "3. Partitioner", "P0 IID · P1 Dirichlet α=0.5\nP3 attack-type m=1 · P4 α=β=0.5\n"
                                   "K = 8 controllers · seeds 0–4", FILL),
        (0, bot, "4. Simulation", "plain PyTorch, clients sequential\nMLP · E=1 · B=256 · plain SGD\n"
                                  "tuned (12-trial random search, seed 99)", FILL2),
        (1, bot, "5. Aggregation (8 strategies)", None, FILL2),
        (2, bot, "6. Metrics + statistics", "val-selected macro-F1, per-class recall\nT95, bytes, worst-client F1\n"
                                            "Friedman/Nemenyi, Wilcoxon, bootstrap,\nregression on JSD / CV", "white"),
    ]
    centers = []
    for col, y, title, body, fc in stages:
        x = x0 + col * (bw + gx)
        box(ax, x, y, bw, bh, fc=fc, lw=0.8)
        text(ax, x + bw / 2, y + bh - 0.15, title, fontweight="bold")
        if body:
            text(ax, x + bw / 2, y + (bh - 0.27) / 2, body, linespacing=1.15)
        centers.append((x, y))
    # strategy names in their colors
    x, y = centers[4]
    rows = [STRATS[:4], STRATS[4:]]
    fracs = ([0.13, 0.37, 0.62, 0.87], [0.13, 0.33, 0.55, 0.83])
    for ri, row in enumerate(rows):
        for fx, s in zip(fracs[ri], row):
            text(ax, x + bw * fx, y + 0.5 - ri * 0.2, LABEL[s], color=TEXT_STRAT[s], fontweight="bold")
    text(ax, x + bw / 2, y + 0.12, "poisoning: f = 2 of 8 (chrono P4, random P3)", color=MAL)
    # arrows along the rows and the turn
    for a, b in ((0, 1), (1, 2), (3, 4), (4, 5)):
        xa, ya = centers[a]
        xb, _ = centers[b]
        arrow(ax, xa + bw, ya + bh / 2, xb, ya + bh / 2)
    xa, ya = centers[2]
    xd, yd = centers[3]
    mid = (ya + yd + bh) / 2
    ax.plot([xa + bw / 2, xa + bw / 2], [ya, mid], color=INK, lw=0.9)
    ax.plot([xa + bw / 2, xd + bw / 2], [mid, mid], color=INK, lw=0.9)
    arrow(ax, xd + bw / 2, mid, xd + bw / 2, yd + bh)
    # protocols band
    text(ax, W / 2, 0.17, "Two evaluation protocols:  chronological split (primary, R = 30)  ·  "
                          "random stratified 80/20 split (secondary, R = 100)", style="italic")
    save(fig, out)


def main() -> None:
    style()
    figd = Path(os.environ.get("FEDSDN_RESULTS", "./results")) / "paper_figures"
    figd.mkdir(parents=True, exist_ok=True)
    fig_system(figd / "fig0_system_architecture.png")
    fig_model(figd / "fig_model_architecture.png")
    fig_pipeline(figd / "fig_pipeline.png")
    print("wrote", ", ".join(p.name for p in sorted(figd.glob("fig*_*.png")) if "architecture" in p.name or "pipeline" in p.name))


if __name__ == "__main__":
    main()
