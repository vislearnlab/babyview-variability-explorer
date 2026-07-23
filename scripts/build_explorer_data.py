#!/usr/bin/env python3
"""Build the explorer payload: per-layer t-SNE layouts + crop thumbnails.

Produces the same columnar-arrays `points.json` shape used by the vislearnlab
drawing explorers, plus a `crops/` directory of small JPEGs.

For each CLS readout (blocks 0-11) and the final projected embedding, we
PCA to 50-d and run t-SNE, giving one 2-D layout per depth. The explorer's
layer slider morphs between them, so you can watch the space reorganize.

Layouts are Procrustes-aligned to the previous layer (rotation/reflection/scale
only, which t-SNE leaves arbitrary) so the morph shows real reorganization
rather than a spurious global spin.

Outputs (explorer/):
  points.json   columnar arrays: category, per-layer x/y, dispersion, file
  crops/        {stem}.jpg thumbnails, long edge THUMB_PX

Run from repo root::

  python scripts/build_explorer_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

REPO = Path(__file__).resolve().parent.parent
LAYERS = REPO / "results" / "clip_layers"
METRICS = REPO / "results" / "metrics"
PUBLIC = REPO / "data" / "valid7018_public"
CCN = REPO / "data" / "ccn2026_results"
OUT = REPO / "explorer"

THUMB_PX = 112
PCA_DIM = 50
SEED = 0
READOUTS = [f"cls_L{i:02d}" for i in range(12)] + ["final"]
LAYER_LABELS = [str(i) for i in range(12)] + ["final"]


def procrustes_align(target: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """Rotate/reflect/scale `moving` onto `target` (t-SNE orientation is arbitrary)."""
    t = target - target.mean(0)
    m = moving - moving.mean(0)
    tn, mn = np.linalg.norm(t), np.linalg.norm(m)
    if tn == 0 or mn == 0:
        return moving
    t, m = t / tn, m / mn
    u, s, vt = np.linalg.svd(m.T @ t)
    r = u @ vt
    return (m @ r) * tn + target.mean(0)


def build_thumbnails(manifest: pd.DataFrame) -> None:
    dest = OUT / "crops"
    dest.mkdir(parents=True, exist_ok=True)
    crop_man = pd.read_csv(PUBLIC / "crops" / "manifest.csv").set_index(["category", "stem"])
    made = 0
    for cat, stem in zip(manifest.category, manifest.stem):
        target = dest / f"{stem}.jpg"
        if target.exists():
            continue
        src = PUBLIC / "crops" / crop_man.loc[(cat, stem), "jpeg_path"]
        im = Image.open(src).convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX), Image.LANCZOS)
        im.save(target, "JPEG", quality=82, optimize=True)
        made += 1
    print(f"thumbnails: {made} written, {len(manifest)} total")


def main() -> None:
    manifest = pd.read_csv(LAYERS / "manifest.csv")
    OUT.mkdir(parents=True, exist_ok=True)
    build_thumbnails(manifest)

    cats = sorted(manifest.category.unique())
    cat_index = {c: i for i, c in enumerate(cats)}
    sem_df = pd.read_csv(CCN / "category_cdi_semantic_map.csv", usecols=["category", "cdi_semantic"]).dropna()
    sem_map = dict(zip(sem_df.category, sem_df.cdi_semantic.str.strip().str.lower()))
    sem_names = sorted({sem_map.get(c, "other") for c in cats})

    # per-layer layouts
    xs, ys = [], []
    prev = None
    for name in READOUTS:
        x = np.load(LAYERS / f"{name}.npy").astype(np.float32)
        x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-10)
        p = PCA(n_components=PCA_DIM, random_state=SEED).fit_transform(x)
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED,
                   learning_rate="auto").fit_transform(p)
        emb = emb - emb.mean(0)
        emb = emb / np.abs(emb).max() * 100.0
        if prev is not None:
            emb = procrustes_align(prev, emb)
        prev = emb
        xs.append(np.round(emb[:, 0], 2).tolist())
        ys.append(np.round(emb[:, 1], 2).tolist())
        print(f"  layout {name}")

    # per-category metrics, per layer
    lm = pd.read_csv(METRICS / "layer_category_metrics.csv")
    gd, ld = {}, {}
    for name in READOUTS:
        d = lm[lm.readout == name].set_index("category")
        gd[name] = [round(float(d.loc[c, "global_dispersion"]), 3) for c in cats]
        ld[name] = [round(float(d.loc[c, "mean_knn_dist"]), 3) for c in cats]

    base = pd.read_csv(METRICS / "baseline_category_metrics.csv")
    pub = base[base.source == "released_clip"].set_index("category")
    pubd = base[base.source == "released_dinov3"].set_index("category")
    freq = pd.read_csv(CCN / "category_frequency_valid85.csv").set_index("category")

    # ---- instance clustering (same-object groups within each category) -----
    # DINOv3 embeddings, average-linkage agglomerative at cos>0.80. Per-crop
    # instance id (contiguous within category) + per-category diversity stats.
    from numpy.linalg import norm
    from sklearn.cluster import AgglomerativeClustering
    INST_COS = 0.80
    dstats = json.loads((PUBLIC / "embedding_norm_stats.json").read_text())["models"]["dinov3"]
    dmu, dsd = np.array(dstats["mu"]), np.array(dstats["sigma"])
    emb_man = pd.read_csv(PUBLIC / "embeddings" / "manifest.csv")
    dkey = emb_man.set_index(["category", "stem"])
    dino = np.stack([np.load(PUBLIC / "embeddings" / dkey.loc[(c, s), "dinov3_npy"]).astype(np.float32) * dsd + dmu
                     for c, s in zip(manifest.category, manifest.stem)])
    dino = dino / np.clip(norm(dino, axis=1, keepdims=True), 1e-8, None)
    inst_per_crop = np.zeros(len(manifest), dtype=int)
    inst_per_crop_ratio, repeat_rate = [], []
    for c in cats:
        idx = np.where(manifest.category.values == c)[0]
        e = dino[idx]
        if len(e) < 2:
            lab = np.zeros(len(e), dtype=int)
        else:
            lab = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - INST_COS,
                                          metric="cosine", linkage="average").fit(e).labels_
        inst_per_crop[idx] = lab
        k = int(lab.max() + 1)
        inst_per_crop_ratio.append(round(k / len(e), 3))
        repeat_rate.append(round(1 - k / len(e), 3))
    print(f"instances: {int(sum((inst_per_crop_ratio[i]*((manifest.category.values==cats[i]).sum())) for i in range(len(cats))))} approx")
    summary = pd.read_csv(METRICS / "layer_summary.csv")
    cls_sum = summary[summary.kind == "cls"].sort_values("block")
    fin = summary[summary.readout == "final"].iloc[0]

    payload = {
        "n": int(len(manifest)),
        "layers": LAYER_LABELS,
        "categories": cats,
        "semantics": sem_names,
        "cat_semantic": [sem_names.index(sem_map.get(c, "other")) for c in cats],
        "crop_dir": "crops",
        "points": {
            "file": [f"{s}.jpg" for s in manifest.stem],
            "cat": [cat_index[c] for c in manifest.category],
            "instance": [int(v) for v in inst_per_crop],   # within-category instance id
            "x": xs,
            "y": ys,
        },
        "category_metrics": {
            "n_exemplars": [int((manifest.category == c).sum()) for c in cats],
            "global_by_layer": [gd[n] for n in READOUTS],
            "local_by_layer": [ld[n] for n in READOUTS],
            "published_global": [round(float(pub.loc[c, "global_dispersion"]), 3) for c in cats],
            "published_local": [round(float(pub.loc[c, "mean_knn_dist"]), 3) for c in cats],
            "dinov3_global": [round(float(pubd.loc[c, "global_dispersion"]), 3) for c in cats],
            "frequency": [float(freq.loc[c, "proportion"]) if c in freq.index else None for c in cats],
            "instances_per_crop": inst_per_crop_ratio,   # 1.0 = every crop a distinct object
            "repeat_rate": repeat_rate,
        },
        "curves": {
            "rho_global": [round(float(v), 4) for v in cls_sum.rho_global_vs_released_clip] +
                          [round(float(fin.rho_global_vs_released_clip), 4)],
            "rho_local": [round(float(v), 4) for v in cls_sum.rho_local_vs_released_clip] +
                         [round(float(fin.rho_local_vs_released_clip), 4)],
            "rho_freq_global": [round(float(v), 4) for v in cls_sum.rho_global_vs_frequency] +
                               [round(float(fin.rho_global_vs_frequency), 4)],
        },
        "meta": {
            "model": "OpenAI CLIP ViT-B/32 (open_clip ViT-B-32-quickgelu)",
            "source": "BabyView valid7018 public release, object-detection@e4883c6",
            "note": "72 privacy-filtered categories; body-part categories withheld.",
            "instance_method": f"DINOv3 agglomerative clustering, cos>{INST_COS}; instance ids are per-category.",
        },
    }
    (OUT / "points.json").write_text(json.dumps(payload, separators=(",", ":")))
    size = (OUT / "points.json").stat().st_size / 1e6
    print(f"wrote {OUT}/points.json ({size:.1f} MB)")


if __name__ == "__main__":
    main()
