#!/usr/bin/env python3
"""Figures for the layer-wise dispersion analyses.

fig1_emergence.png  -- rank agreement with the published final-layer metrics,
                       as a function of CLIP block, for both readouts
fig2_frequency.png  -- frequency/dispersion coupling by depth (the paper's
                       weak effect, tested at every layer)

Run from repo root::

  python scripts/plot_layer_curves.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "results" / "metrics"
FIGS = REPO / "results" / "figures"

CLS_C = "#2B6CB0"
MEAN_C = "#C05621"
DINO_C = "#2F855A"


def block_series(s: pd.DataFrame, kind: str, col: str):
    d = s[s.kind == kind].sort_values("block")
    return d.block, d[col]


def main() -> None:
    s = pd.read_csv(METRICS / "layer_summary.csv")
    FIGS.mkdir(parents=True, exist_ok=True)
    final = s[s.readout == "final"].iloc[0]

    # ---- Figure 1: emergence -------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    panels = [
        ("global_dispersion", "rho_global_vs_released_clip", "rho_global_vs_released_dinov3",
         "Global dispersion (distance to centroid)"),
        ("mean_knn_dist", "rho_local_vs_released_clip", "rho_local_vs_released_dinov3",
         "Local dispersion (kNN, k=5)"),
    ]
    for ax, (_, clip_col, dino_col, title) in zip(axes, panels):
        for kind, color, lbl in [("cls", CLS_C, "CLS token"), ("mean", MEAN_C, "mean patch tokens")]:
            x, y = block_series(s, kind, clip_col)
            ax.plot(x, y, "o-", color=color, lw=2, ms=5, label=f"{lbl} vs published CLIP")
            x, y = block_series(s, kind, dino_col)
            ax.plot(x, y, "s--", color=color, lw=1.2, ms=4, alpha=0.55,
                    label=f"{lbl} vs published DINOv3")
        ax.axhline(final[clip_col], color=CLS_C, ls=":", lw=1, alpha=0.7)
        ax.scatter([11.6], [final[clip_col]], marker="*", s=160, color=CLS_C,
                   zorder=5, label="final projected (paper's readout)")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("CLIP ViT-B/32 transformer block")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(0, 12))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Spearman rho across 72 categories")
    axes[1].legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.suptitle(
        "Where does the category-dispersion ordering come from? (72 public categories, N=5,921)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_emergence.png", dpi=200)
    fig.savefig(FIGS / "fig1_emergence.pdf")

    # ---- Figure 2: frequency coupling by depth --------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for kind, color, lbl in [("cls", CLS_C, "CLS token"), ("mean", MEAN_C, "mean patch tokens")]:
        x, y = block_series(s, kind, "rho_global_vs_frequency")
        ax.plot(x, y, "o-", color=color, lw=2, ms=5, label=f"{lbl} (global)")
        x, y = block_series(s, kind, "rho_local_vs_frequency")
        ax.plot(x, y, "^--", color=color, lw=1.2, ms=4, alpha=0.55, label=f"{lbl} (local)")
    ax.scatter([11.6], [final["rho_global_vs_frequency"]], marker="*", s=160,
               color=CLS_C, zorder=5, label="final projected")
    ax.axhline(0, color="#999", lw=0.8)
    ax.axhspan(-0.23, 0.23, color="#bbb", alpha=0.18, zorder=0)
    ax.text(0.1, 0.235, "|rho| < .23  (p > .05, n=72)", fontsize=7.5, color="#555")
    ax.set_xlabel("CLIP ViT-B/32 transformer block")
    ax.set_ylabel("Spearman rho with category frequency")
    ax.set_title("Frequency-dispersion coupling is weak at every depth", fontsize=11)
    ax.set_xticks(range(0, 12))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_frequency.png", dpi=200)
    fig.savefig(FIGS / "fig2_frequency.pdf")
    print(f"wrote {FIGS}")


if __name__ == "__main__":
    main()
