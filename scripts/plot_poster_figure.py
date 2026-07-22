#!/usr/bin/env python3
"""Poster figure for the layer-wise dispersion results.

Follows the CCN 2026 figure conventions from
`analysis/ccn-2026/scripts/generate_valid7018_paper_figures.py`:
the CDI_SEMANTIC_COLORS palette, bold axis labels at 11pt, bold titles at 12pt,
white rounded annotation boxes, light grid.

Panels:
  A  rank agreement with the published final-layer ordering, by block
     (global vs local -- the dissociation)
  B  block 0 vs final global dispersion, colored by CDI semantic
  C  block 0 vs final local dispersion, same coloring
  D  frequency-dispersion coupling at every depth (the null)

Run from repo root::

  python scripts/plot_poster_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "results" / "metrics"
CCN = REPO / "data" / "ccn2026_results"
FIGS = REPO / "results" / "figures"

# --- verbatim from generate_valid7018_paper_figures.py ---------------------
CDI_SEMANTIC_COLORS = {
    "animals": "#4DB8A8",
    "body_parts": "#E87A5F",
    "clothing": "#9B7EC8",
    "food_drink": "#E8A54C",
    "furniture_rooms": "#6BAB7A",
    "household": "#D97B9E",
    "outside": "#5B9BD5",
    "people": "#E8C44C",
    "toys": "#B07CC8",
    "vehicles": "#6BA3D5",
    "other": "#8B9A9E",
}
CDI_SEMANTIC_ORDER = [
    "animals", "body_parts", "clothing", "food_drink", "furniture_rooms",
    "household", "outside", "people", "toys", "vehicles", "other",
]

GLOBAL_C = "#1F4E79"
LOCAL_C = "#C0561F"
BOX = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#BBBBBB", alpha=0.92)


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def panel_letter(ax, letter):
    ax.text(-0.13, 1.06, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", ha="left")


def semantic_map() -> dict[str, str]:
    df = pd.read_csv(CCN / "category_cdi_semantic_map.csv", usecols=["category", "cdi_semantic"]).dropna()
    df["cdi_semantic"] = df["cdi_semantic"].astype(str).str.strip().str.lower()
    return dict(zip(df["category"], df["cdi_semantic"]))


def scatter_panel(ax, layers, baseline, sem, col, title, ylabel):
    # y-axis is the *published* CLIP value, so these panels are the block-0
    # endpoints of panel A rather than a separate comparison.
    early = layers[layers.readout == "cls_L00"].set_index("category")[col]
    late = baseline[baseline.source == "released_clip"].set_index("category")[col]
    cats = early.index.intersection(late.index)
    colors = [CDI_SEMANTIC_COLORS.get(sem.get(c, "other"), CDI_SEMANTIC_COLORS["other"]) for c in cats]
    ax.scatter(early[cats], late[cats], c=colors, s=46, alpha=0.9,
               edgecolors="white", linewidths=0.6, zorder=3)
    rho, p = spearmanr(early[cats], late[cats])
    ptxt = "p < .001" if p < 1e-3 else f"p = {p:.3f}"
    ax.text(0.03, 0.96, f"rho = {rho:.2f}, {ptxt}", transform=ax.transAxes,
            fontsize=10, va="top", bbox=BOX)
    style_axes(ax, title, "Block 0 (early, low-level)", ylabel)


def main() -> None:
    s = pd.read_csv(METRICS / "layer_summary.csv")
    layers = pd.read_csv(METRICS / "layer_category_metrics.csv")
    baseline = pd.read_csv(METRICS / "baseline_category_metrics.csv")
    freq = pd.read_csv(CCN / "category_frequency_valid85.csv")
    sem = semantic_map()
    FIGS.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.32,
                          left=0.07, right=0.985, top=0.88, bottom=0.09)

    cls = s[s.kind == "cls"].sort_values("block")
    final = s[s.readout == "final"].iloc[0]

    # ---- A: emergence ----------------------------------------------------
    axA = fig.add_subplot(gs[0, :2])
    axA.plot(cls.block, cls.rho_global_vs_released_clip, "o-", color=GLOBAL_C,
             lw=2.6, ms=7, label="Global dispersion (distance to centroid)", zorder=4)
    axA.plot(cls.block, cls.rho_local_vs_released_clip, "s--", color=LOCAL_C,
             lw=2.2, ms=6, label="Local dispersion (kNN, k = 5)", zorder=4)
    axA.scatter([12], [final.rho_global_vs_released_clip], marker="*", s=280,
                color=GLOBAL_C, zorder=6, edgecolors="white", linewidths=0.8)
    axA.scatter([12], [final.rho_local_vs_released_clip], marker="*", s=280,
                color=LOCAL_C, zorder=6, edgecolors="white", linewidths=0.8)
    axA.axvspan(9.5, 12.4, color="#F0C808", alpha=0.13, zorder=0)
    axA.text(11.0, 0.10, "ordering built\nhere", fontsize=9.5, ha="center",
             color="#6B5800", fontweight="bold")
    axA.annotate("published readout", xy=(12, final.rho_global_vs_released_clip),
                 xytext=(8.4, 1.03), fontsize=9, color="#141414", bbox=BOX,
                 arrowprops=dict(arrowstyle="-", color="#666666", lw=0.8))
    axA.axhline(0, color="#999", lw=0.8)
    axA.set_xticks(list(range(12)) + [12])
    axA.set_xticklabels([str(i) for i in range(12)] + ["final"])
    axA.set_ylim(-0.08, 1.14)
    style_axes(axA, "Global dispersion is built late; local dispersion is not",
               "CLIP ViT-B/32 transformer block", "Spearman rho with published\nfinal-layer ordering")
    axA.legend(fontsize=10, loc="lower left", framealpha=0.95, edgecolor="#BBBBBB")
    panel_letter(axA, "A")

    # ---- D: frequency null ----------------------------------------------
    axD = fig.add_subplot(gs[0, 2])
    axD.axhspan(-0.232, 0.232, color="#8B9A9E", alpha=0.16, zorder=0)
    axD.plot(cls.block, cls.rho_global_vs_frequency, "o-", color=GLOBAL_C, lw=2.2, ms=6,
             label="Global", zorder=3)
    axD.plot(cls.block, cls.rho_local_vs_frequency, "s--", color=LOCAL_C, lw=1.9, ms=5,
             label="Local", zorder=3)
    axD.scatter([12], [final.rho_global_vs_frequency], marker="*", s=230, color=GLOBAL_C,
                zorder=5, edgecolors="white", linewidths=0.8)
    axD.axhline(0, color="#999", lw=0.8)
    axD.set_xticks([0, 3, 6, 9, 12])
    axD.set_xticklabels(["0", "3", "6", "9", "final"])
    axD.set_ylim(-0.55, 0.55)
    axD.text(0.5, 0.055, "n.s. band (p > .05)", transform=axD.transAxes, fontsize=8.5,
             ha="center", color="#52514E", style="italic")
    style_axes(axD, "Frequency coupling:\nweak at every depth",
               "Transformer block", "rho with category frequency")
    axD.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor="#BBBBBB")
    panel_letter(axD, "D")

    # ---- B, C: early vs final scatters ------------------------------------
    axB = fig.add_subplot(gs[1, 0])
    scatter_panel(axB, layers, baseline, sem, "global_dispersion",
                  "Global: early does not predict final", "Published final-layer value")
    panel_letter(axB, "B")

    axC = fig.add_subplot(gs[1, 1])
    scatter_panel(axC, layers, baseline, sem, "mean_knn_dist",
                  "Local: early already predicts final", "Published final-layer value")
    panel_letter(axC, "C")

    # ---- legend panel -----------------------------------------------------
    axL = fig.add_subplot(gs[1, 2])
    axL.axis("off")
    present = [g for g in CDI_SEMANTIC_ORDER if g in {sem.get(c, "other") for c in layers.category.unique()}]
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=CDI_SEMANTIC_COLORS[g],
                      markeredgecolor="none", markersize=9, label=g.replace("_", " "))
               for g in present]
    leg = axL.legend(handles=handles, title="CDI semantic category", loc="upper center",
                     ncol=2, fontsize=10, title_fontsize=11, frameon=True,
                     edgecolor="#BBBBBB", bbox_to_anchor=(0.5, 1.02))
    leg.get_title().set_fontweight("bold")
    axL.text(0.5, 0.30, "72 categories, N = 5,921 validated crops\n"
                        "Public privacy-filtered release\n"
                        "(body-part categories withheld)\n\n"
                        "Metrics computed with the CCN 2026\n"
                        "pipeline, unchanged, at every block",
             transform=axL.transAxes, ha="center", va="top", fontsize=9.5, color="#52514E")

    fig.suptitle("Where in CLIP does everyday object-category variability arise?",
                 fontsize=15.5, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925, "Layer-wise reanalysis of the CCN 2026 BabyView dispersion metrics",
             ha="center", fontsize=11.5, color="#52514E")

    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"poster_layerwise.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote {FIGS}/poster_layerwise.png / .pdf")


if __name__ == "__main__":
    main()
