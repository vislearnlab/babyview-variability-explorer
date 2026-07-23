#!/usr/bin/env python3
"""Group crops by individual exemplar (same physical object) within each category.

Motivation: the CCN abstract measures category dispersion but notes it "does not
distinguish between different sources of exemplar variability" -- e.g. one cup
seen from many angles vs. many different cups. This script takes a first cut at
that decomposition by clustering the crops within each category into estimated
object instances, then asks whether a category's dispersion reflects how many
DISTINCT objects it contains.

Method. DINOv3 embeddings discriminate instances well (CLIP saturates near cos
0.91 for everything and is useless here). Within each category we run
average-linkage agglomerative clustering on cosine distance with a fixed
threshold; each cluster is an estimated instance. Repeated near-duplicate /
similar-view crops of one object merge; genuinely different objects stay apart.

STRONG CAVEATS (read before trusting):
  * No ground truth and no temporal/video linkage (the public release uses
    opaque stems), so this is unvalidated and threshold-dependent.
  * Embedding cosine captures same-object-similar-view best; the same object
    from a very different angle can exceed the threshold and split into two
    "instances". So instance counts are an UPPER bound and repeat-rates a LOWER
    bound. Eyeball the montages before believing any number.

Outputs (results/instances/):
  instance_counts.csv    per category: n_crops, n_instances, repeat_rate, ...
  instance_dispersion.csv correlation of instance-diversity with dispersion
  montages/{cat}.jpg      crops grouped by estimated instance (rows = instances)

Run from repo root::

  python scripts/cluster_instances.py --threshold 0.80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from PIL import Image
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "data" / "valid7018_public"
METRICS = REPO / "results" / "metrics"
OUT = REPO / "results" / "instances"

MONTAGE_CATS = ["clock", "oven", "sink", "book", "car", "ball"]  # low- vs high-dispersion
CELL = 96  # montage thumbnail px


def load_dino() -> tuple[pd.DataFrame, np.ndarray]:
    man = pd.read_csv(PUBLIC / "embeddings" / "manifest.csv")
    st = json.loads((PUBLIC / "embedding_norm_stats.json").read_text())["models"]["dinov3"]
    mu, sd = np.array(st["mu"]), np.array(st["sigma"])
    emb = np.stack([np.load(PUBLIC / "embeddings" / p).astype(np.float32) * sd + mu
                    for p in man.dinov3_npy])
    emb = emb / np.clip(norm(emb, axis=1, keepdims=True), 1e-8, None)
    return man, emb


def cluster_category(e: np.ndarray, cos_thresh: float) -> np.ndarray:
    if len(e) < 2:
        return np.zeros(len(e), dtype=int)
    cl = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - cos_thresh,
        metric="cosine", linkage="average").fit(e)
    return cl.labels_


def montage(cat: str, man: pd.DataFrame, labels: np.ndarray, crop_man: pd.DataFrame, dest: Path):
    """Rows = estimated instances (multi-crop first), up to a cap; cols = crops."""
    idx = man.index[man.category == cat].to_numpy()
    sizes = np.bincount(labels)
    order = np.argsort(-sizes)  # biggest instance groups first
    rows = [np.where(labels == c)[0] for c in order]
    rows = [r for r in rows if len(r) >= 1]
    max_rows, max_cols = 14, 10
    rows = rows[:max_rows]
    ncol = min(max_cols, max((len(r) for r in rows), default=1))
    W, H = ncol * CELL, len(rows) * CELL
    if W == 0 or H == 0:
        return
    canvas = Image.new("RGB", (W, H), (245, 245, 243))
    cm = crop_man.set_index(["category", "stem"])
    for ri, r in enumerate(rows):
        for ci, member in enumerate(r[:max_cols]):
            stem = man.loc[idx[member], "stem"]
            jp = cm.loc[(cat, stem), "jpeg_path"]
            im = Image.open(PUBLIC / "crops" / jp).convert("RGB")
            im.thumbnail((CELL, CELL), Image.LANCZOS)
            off = ((CELL - im.width) // 2, (CELL - im.height) // 2)
            canvas.paste(im, (ci * CELL + off[0], ri * CELL + off[1]))
    dest.mkdir(parents=True, exist_ok=True)
    canvas.save(dest / f"{cat}.jpg", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.80,
                    help="cosine similarity to merge into one instance (higher = stricter)")
    ap.add_argument("--montages", type=int, default=1, help="render montages (1/0)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    man, emb = load_dino()
    crop_man = pd.read_csv(PUBLIC / "crops" / "manifest.csv")
    cats = sorted(man.category.unique())

    rows = []
    labels_by_cat = {}
    for c in cats:
        idx = man.index[man.category == c].to_numpy()
        e = emb[idx]
        lab = cluster_category(e, args.threshold)
        labels_by_cat[c] = lab
        n = len(e)
        k = int(lab.max() + 1)
        sizes = np.bincount(lab)
        rows.append({
            "category": c,
            "n_crops": n,
            "n_instances": k,
            "instances_per_crop": k / n,          # 1.0 = every crop distinct
            "repeat_rate": 1 - k / n,             # fraction of crops that are repeats
            "n_multi_instances": int((sizes >= 2).sum()),
            "largest_instance": int(sizes.max()),
            "pct_crops_in_repeats": float((sizes[sizes >= 2].sum()) / n),
        })
    counts = pd.DataFrame(rows)
    counts.to_csv(OUT / "instance_counts.csv", index=False)

    # ---- relate instance diversity to the paper's dispersion ---------------
    base = pd.read_csv(METRICS / "baseline_category_metrics.csv")
    dino = base[base.source == "released_dinov3"].set_index("category")
    clip = base[base.source == "released_clip"].set_index("category")
    m = counts.set_index("category")
    common = m.index.intersection(dino.index)
    res_rows = []
    for disp_name, disp in [("dinov3_global", dino.global_dispersion),
                            ("dinov3_local", dino.mean_knn_dist),
                            ("clip_global", clip.global_dispersion),
                            ("clip_local", clip.mean_knn_dist)]:
        for div_name in ["instances_per_crop", "repeat_rate", "pct_crops_in_repeats"]:
            rho, p = spearmanr(m.loc[common, div_name], disp.loc[common])
            res_rows.append({"dispersion": disp_name, "diversity": div_name,
                             "spearman_rho": float(rho), "p_value": float(p),
                             "n_categories": len(common)})
    res = pd.DataFrame(res_rows)
    res.to_csv(OUT / "instance_dispersion.csv", index=False)

    print(f"threshold cos>{args.threshold}: "
          f"{counts.n_instances.sum()} instances across {counts.n_crops.sum()} crops")
    print("\nmost repeated (lowest instances_per_crop):")
    print(counts.nsmallest(6, "instances_per_crop")[
        ["category", "n_crops", "n_instances", "repeat_rate", "largest_instance"]].to_string(index=False))
    print("\nmost distinct (highest instances_per_crop):")
    print(counts.nlargest(6, "instances_per_crop")[
        ["category", "n_crops", "n_instances", "repeat_rate"]].to_string(index=False))
    print("\n=== does dispersion track number of DISTINCT instances? ===")
    key = res[(res.diversity == "instances_per_crop")]
    for _, r in key.iterrows():
        print(f"  {r.dispersion:14s} vs instances_per_crop: rho={r.spearman_rho:+.3f} (p={r.p_value:.3g})")

    (OUT / "instance_run.json").write_text(json.dumps({
        "threshold_cosine": args.threshold,
        "features": "DINOv3-final (released), L2-normalized, average-linkage agglomerative",
        "n_categories": len(cats),
        "total_crops": int(counts.n_crops.sum()),
        "total_instances": int(counts.n_instances.sum()),
        "caveat": "no ground truth / no temporal linkage; instance counts are an upper bound. "
                  "Eyeball montages before trusting.",
    }, indent=2) + "\n")

    if args.montages:
        for c in MONTAGE_CATS:
            if c in labels_by_cat:
                montage(c, man, labels_by_cat[c], crop_man, OUT / "montages")
        print(f"\nmontages -> {OUT/'montages'}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
