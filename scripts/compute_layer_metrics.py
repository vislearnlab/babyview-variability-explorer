#!/usr/bin/env python3
"""Recompute the CCN 2026 dispersion metrics at every CLIP layer.

For each readout (cls_L00..cls_L11, mean_L00..mean_L11, final) we:
  1. feature-wise z-score, mu/sigma fit on all 5,921 public crops pooled across
     the 72 categories -- the same normalization *form* as the paper, refit on
     the public cohort (the paper's mu/sigma were fit on all 7,018);
  2. run the paper's compute_category_metrics() unchanged, k=5, Euclidean.

Also recomputes the same metrics on the *released* CLIP and DINOv3 vectors
restricted to the same 72 categories, so every comparison is apples-to-apples
on one cohort.

Because Euclidean distance in d z-scored dimensions grows like sqrt(d), and the
block readouts are 768-d while `final` is 512-d, absolute dispersion is NOT
comparable across readouts. We emit a `*_per_sqrt_dim` column for rough scale
comparison, but all inference should rest on across-category rank correlations.

Outputs (results/metrics/):
  layer_category_metrics.csv     readout x category x metrics
  baseline_category_metrics.csv  released clip + dinov3, same 72 categories
  layer_summary.csv              per-readout Spearman vs baselines and frequency
  metrics_run.json               provenance

Run from repo root::

  python scripts/compute_layer_metrics.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from category_metrics import compute_category_metrics

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "data" / "valid7018_public"
CCN = REPO / "data" / "ccn2026_results"
# LAYERS / OUT are set in main() from the --model flag (clip or dinov3), so the
# same code produces both models' layer metrics into separate result dirs.
LAYERS = REPO / "results" / "clip_layers"
OUT = REPO / "results" / "metrics"

K = 5
STATS_EPS = 1e-10  # matches valid7018_embedding_normalize.py
N_BLOCKS = 12


def readout_names() -> list[str]:
    names = [f"cls_L{i:02d}" for i in range(N_BLOCKS)]
    names += [f"mean_L{i:02d}" for i in range(N_BLOCKS)]
    names += ["final"]
    return names


def cohort_zscore(x: np.ndarray) -> np.ndarray:
    """Feature-wise z-score fit on the pooled cohort (ddof=0), as in the paper."""
    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, ddof=0, keepdims=True)
    return (x - mu) / np.maximum(sigma, STATS_EPS)


def by_category(x: np.ndarray, cats: pd.Series) -> dict[str, np.ndarray]:
    return {c: x[np.asarray(cats == c)] for c in sorted(cats.unique())}


def load_released(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Released CLIP + DINOv3 vectors, in manifest row order."""
    emb_man = pd.read_csv(PUBLIC / "embeddings" / "manifest.csv")
    key = emb_man.set_index(["category", "stem"])
    out = {}
    for model, col in [("clip", "clip_npy"), ("dinov3", "dinov3_npy")]:
        rows = [
            np.load(PUBLIC / "embeddings" / key.loc[(c, s), col]).astype(np.float32)
            for c, s in zip(manifest.category, manifest.stem)
        ]
        out[model] = np.stack(rows)
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["clip", "dinov3"], default="clip",
                    help="which extracted layer set to score")
    args = ap.parse_args()
    global LAYERS, OUT
    LAYERS = REPO / "results" / f"{args.model}_layers"
    OUT = REPO / "results" / ("metrics" if args.model == "clip" else f"metrics_{args.model}")

    manifest = pd.read_csv(LAYERS / "manifest.csv")
    cats = manifest.category
    n_cat = cats.nunique()
    print(f"[{args.model}] {len(manifest)} exemplars, {n_cat} categories -> {OUT.name}")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- baselines: released vectors, same cohort -------------------------
    released = load_released(manifest)
    base_frames = []
    for model, vecs in released.items():
        # Already z-scored with the 7,018-crop mu/sigma as shipped; we do NOT
        # refit, so these reproduce the published numbers up to the 72-cat subset.
        df = compute_category_metrics(by_category(vecs.astype(np.float64), cats), K)
        df.insert(0, "source", f"released_{model}")
        base_frames.append(df)
    baseline = pd.concat(base_frames, ignore_index=True)
    baseline.to_csv(OUT / "baseline_category_metrics.csv", index=False)
    print(f"baselines written ({len(baseline)} rows)")

    # ---- layer-wise -------------------------------------------------------
    frames = []
    for name in readout_names():
        x = np.load(LAYERS / f"{name}.npy").astype(np.float64)
        z = cohort_zscore(x)
        df = compute_category_metrics(by_category(z, cats), K)
        df.insert(0, "readout", name)
        df.insert(1, "kind", "final" if name == "final" else name.split("_")[0])
        df.insert(2, "block", -1 if name == "final" else int(name.split("_L")[1]))
        df.insert(3, "dim", x.shape[1])
        df["global_dispersion_per_sqrt_dim"] = df.global_dispersion / np.sqrt(x.shape[1])
        df["mean_knn_dist_per_sqrt_dim"] = df.mean_knn_dist / np.sqrt(x.shape[1])
        frames.append(df)
        print(f"  {name:10s} dim={x.shape[1]:4d}  mean V_c={df.global_dispersion.mean():.2f}")
    layers = pd.concat(frames, ignore_index=True)
    layers.to_csv(OUT / "layer_category_metrics.csv", index=False)

    # ---- summary: how each readout relates to the published quantities ----
    freq = pd.read_csv(CCN / "category_frequency_valid85.csv")[
        ["category", "proportion", "cdi_semantic"]
    ]
    ref = {
        m: baseline[baseline.source == f"released_{m}"].set_index("category")
        for m in ("clip", "dinov3")
    }

    rows = []
    for name, df in layers.groupby("readout", sort=False):
        d = df.set_index("category")
        common = d.index.intersection(ref["clip"].index)
        f = freq.set_index("category").reindex(common)
        rec: dict = {
            "readout": name,
            "kind": df.kind.iloc[0],
            "block": int(df.block.iloc[0]),
            "dim": int(df.dim.iloc[0]),
            "n_categories": len(common),
            "mean_global_dispersion": float(d.global_dispersion.mean()),
            "sd_global_dispersion": float(d.global_dispersion.std(ddof=1)),
            "mean_knn_dist": float(d.mean_knn_dist.mean()),
        }
        for metric, mine_col, ref_col in [
            ("global", "global_dispersion", "global_dispersion"),
            ("local", "mean_knn_dist", "mean_knn_dist"),
        ]:
            for model in ("clip", "dinov3"):
                rho, p = spearmanr(d.loc[common, mine_col], ref[model].loc[common, ref_col])
                rec[f"rho_{metric}_vs_released_{model}"] = float(rho)
                rec[f"p_{metric}_vs_released_{model}"] = float(p)
            rho, p = spearmanr(d.loc[common, mine_col], f.proportion)
            rec[f"rho_{metric}_vs_frequency"] = float(rho)
            rec[f"p_{metric}_vs_frequency"] = float(p)
        # within-readout global vs local coupling
        rho, p = spearmanr(d.global_dispersion, d.mean_knn_dist)
        rec["rho_global_vs_local_within"] = float(rho)
        rec["p_global_vs_local_within"] = float(p)
        rows.append(rec)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "layer_summary.csv", index=False)

    (OUT / "metrics_run.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "k": K,
                "n_exemplars": int(len(manifest)),
                "n_categories": int(n_cat),
                "metric": "euclidean",
                "layer_normalization": "featurewise_zscore_within_valid7018_public_cohort_5921",
                "baseline_normalization": "as-shipped (featurewise z-score fit on full 7,018)",
                "metric_code": "verbatim copy of analysis/ccn-2026/scripts/valid7018_category_metrics.py @ e4883c6",
                "caveat_dimensionality": (
                    "Euclidean distance scales ~sqrt(d); block readouts are 768-d vs 512-d for "
                    "`final`. Compare readouts via across-category rank correlations, not raw V_c."
                ),
                "caveat_cohort": (
                    "72 public categories only; body parts and glasses are excluded from the "
                    "public release for privacy. Paper reports 85."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
