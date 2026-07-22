# BabyView CLIP Layer Explorer

Interactive page for the layer-wise reanalysis of the CCN 2026 variability
abstract. Built in the style of the
[`drawing-explorer`](https://github.com/vislearnlab/drawing-explorer) /
[`sea-animals-draw-explorer`](https://github.com/vislearnlab/sea-animals-draw-explorer)
pages: self-contained HTML, columnar `points.json`, canvas rendering, no build
step at serve time.

## What it shows

**5,921 human-validated object crops** across **72 CDI noun categories**, from
the public privacy-filtered BabyView release. Each point is one crop, laid out by
t-SNE of its CLIP ViT-B/32 representation **at a chosen transformer block**.

The centerpiece is the **layer slider**. Drag it (or hit Play) to move from block
0 to the final projected embedding and watch the space reorganize — the visual
counterpart of the result that global dispersion is constructed in the last two
blocks rather than inherited from low-level image statistics.

- **Color by** CDI semantic group (paper palette), global dispersion, or local
  dispersion. The dispersion scales recompute per layer.
- **Filter** by semantic group and/or category; unmatched points stay as faint
  context.
- **Dots / Crops** toggle renders the actual crop thumbnails in place of points.
- **Hover** a point for the crop and its category's dispersion *at the current
  block*; click to pin.
- The right column tracks agreement with the published final-layer ordering as
  you move through depth, and — with a single category selected — that category's
  dispersion trajectory across all 13 readouts.

## Run locally

```bash
python3 -m http.server 8000    # from this directory
# open http://127.0.0.1:8000/
```

## Files

- `index.html` — the whole explorer, self-contained.
- `points.json` (3.0 MB) — columnar payload: 13 t-SNE layouts, per-category
  metrics per layer, published CLIP/DINOv3 values, summary curves.
- `crops/` (24 MB) — 5,921 thumbnails, long edge 112 px.
- Rebuild both with `python ../scripts/build_explorer_data.py`.

## Notes on the layouts

Each layer's layout is an independent t-SNE (PCA to 50-d first, perplexity 30,
seed 0). t-SNE fixes neither orientation nor scale, so each layout is
**Procrustes-aligned** (rotation/reflection/scale only) to the previous block —
otherwise the morph would show an arbitrary global spin instead of real
reorganization. Positions are still not comparable *across* layers in any metric
sense; the quantitative claims all come from the rank correlations in
`results/metrics/`, not from the visual layout.

Category dispersion values are computed with the CCN 2026 pipeline unchanged
(Euclidean, k = 5, cohort z-score).
