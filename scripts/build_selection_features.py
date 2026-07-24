#!/usr/bin/env python3
"""Compact per-crop features for on-demand t-SNE recompute in the explorer.

The explorer's precomputed layouts are 2-D only. To let the page re-run t-SNE on
a user-selected subset of categories, the browser needs the *high-dimensional*
features. Shipping every layer's full 768-d vectors would be huge, so we ship a
PCA-reduced version of each model's **final** embedding — enough to re-embed a
handful of categories with structure intact, at a few MB.

Output: explorer/features.json
  {dim, models:[...], features:{clip:[[..dim..] x 5921], dinov3:[...]}}
Rows align 1:1 with explorer/points.json (same manifest order).

Run from repo root::

  python scripts/build_selection_features.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "explorer" / "features.json"
PCA_DIM = 30
SEED = 0


def cohort_zscore(x):
    return (x - x.mean(0)) / np.maximum(x.std(0), 1e-10)


def main():
    # align to the explorer manifest (clip layers manifest == the shared order)
    manifest = pd.read_csv(REPO / "results" / "clip_layers" / "manifest.csv")
    n = len(manifest)
    models = {"clip": REPO / "results" / "clip_layers" / "final.npy"}
    if (REPO / "results" / "dinov3_layers" / "final.npy").exists():
        models["dinov3"] = REPO / "results" / "dinov3_layers" / "final.npy"

    feats = {}
    for name, path in models.items():
        x = np.load(path).astype(np.float32)
        assert len(x) == n, f"{name} rows {len(x)} != manifest {n}"
        z = cohort_zscore(x)
        p = PCA(n_components=PCA_DIM, random_state=SEED).fit_transform(z)
        # round to 3 decimals to keep the file small; PCA components are ~unit-ish
        feats[name] = np.round(p, 3).tolist()
        print(f"{name}: PCA {x.shape[1]} -> {PCA_DIM}")

    OUT.write_text(json.dumps({
        "dim": PCA_DIM,
        "models": list(feats.keys()),
        "note": "PCA of each model's final embedding (cohort z-scored). Rows align with points.json.",
        "features": feats,
    }, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
