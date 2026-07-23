# CLAUDE.md — babyview-variability-explorer

Context for working in this repo. Read alongside `README.md` and `NOTE_FOR_JANE.md`.

## What this is

Exploratory follow-up to the CCN 2026 abstract *"Variability in Young Children's
Everyday Visual Experiences of Object Categories"* (Yang, Sepuri, Tan, Aw, Frank
& Long). The abstract measured object-category dispersion in the **final** CLIP
embedding; this repo recomputes the same metrics at **every one of the 12 CLIP
ViT-B/32 transformer blocks** and ships an interactive explorer.

- Local dir: `~/Documents/GitHub/object-detection-clip-layers`
- GitHub: `vislearnlab/babyview-variability-explorer` (public)
- Live (poster URL): https://vislearnlab.github.io/babyview-variability-explorer/

Separate from the upstream `object-detection` repo on purpose — do **not** treat
this as a branch of it. The local `object-detection` clone has unstaged WIP;
leave it alone.

## Data provenance

Everything derives from the **public, privacy-filtered** release in
`babyview-project/object-detection` @ commit `e4883c6`
(`data/shared_data_ccn_2026/public/`). No restricted BabyView data is used.

- **5,921 crops, 72 categories** — the paper's 85 minus 13 body-part/glasses
  categories withheld for privacy. Run analyses on these 72 as-is; do **not**
  go looking for the withheld crops.
- Model is OpenAI **`ViT-B-32-quickgelu`** (open_clip). Confirmed empirically:
  re-extracting the final layer reproduces the released vectors at r = .985;
  LAION-2B weights give r ≈ .00. The upstream repo never recorded the checkpoint.

## Pipeline

```
python scripts/extract_clip_layers.py     # 12 blocks × 5,921 crops → results/clip_layers/ (~4 min, MPS)
python scripts/compute_layer_metrics.py   # dispersion per readout → results/metrics/
python scripts/plot_layer_curves.py       # fig1/fig2
python scripts/plot_poster_figure.py      # results/figures/poster_layerwise.{png,pdf}
python scripts/build_explorer_data.py     # explorer/points.json + explorer/crops/
```

`scripts/category_metrics.py` is a **verbatim copy** of the paper's
`valid7018_category_metrics.py` — keep it byte-identical so layer numbers come
from the same code as the published ones. The `final` readout reproduces the
published per-category values at rho = .993 (global) / .994 (local).

Before running scripts, unzip the archives (they are gitignored when unzipped):
`cd data/valid7018_public && unzip -o crops.zip -d crops && unzip -o embeddings.zip -d embeddings`.

## Key results (all rank-based)

Euclidean distance scales like √d and readouts differ in dimensionality, so
**every claim rests on across-category rank correlations, never raw magnitudes.**

- **Global dispersion ordering is built late** (rho .04 at block 8 → .93 at
  block 11 → .99 final); **local kNN ordering is present early** (rho .65 at
  block 0). This dissociation is the headline.
- Frequency–dispersion coupling is weak at **every** depth (|rho| < .24) — the
  paper's null is not a readout artifact.
- CLIP mean-patch readouts match DINOv3 (.60–.62) better than CLIP's own CLS
  (.52) — cross-model agreement is partly a readout mismatch.
- Dispersion does **not** track dimensionality: the cohort z-score pins scale so
  mean V_c ≈ √d regardless of representation; effective dim still climbs 10→70
  with depth, sharpest at blocks 9–11.

## The stale-export finding (do not misread)

The upstream `valid7018` metrics live in two places and disagree.
`analysis/ccn-2026/valid7018/` is correct and matches the abstract exactly
(CLIP global mean 18.12, DINOv3 23.36). `data/shared_data_ccn_2026/valid7018/`
is a **stale export** (24.46 / 32.08) — commit `f535af9` regenerated only the
analysis copy after switching normalization. Ranks unaffected (r = .9999); **no
paper conclusion changes.** When comparing, read the `analysis/` copy. See
`NOTE_FOR_JANE.md`.

## Explorer conventions (`explorer/index.html`)

Built in the vislearnlab drawing-explorer house style: self-contained single
HTML file, columnar `points.json`, `<canvas>` rendering with a lazy thumbnail
cache, theme-aware (light/dark via `data-theme` + `prefers-color-scheme`), no
emoji, no external dependencies.

- Layer slider defaults to the **final** embedding; drag left toward block 0.
- Color modes: CDI semantic, **Category** (per-category hues derived from the
  category's CDI group base color so a group reads as one color family),
  global/local dispersion. Crops view with optional colored borders.
- Selected-categories panel is a **scatter** (freq×global / freq×local /
  global×local — the paper's Fig 1C framing). Bars were tried and rejected as
  uninformative — do not reintroduce single-metric normalized bars.
- Scroll to zoom (cursor-anchored, 1–40×), drag to pan, double-click to reset.

### Verifying explorer edits

The Chrome extension was not connected during development, so the UI has been
validated only structurally, never seen rendered. After any edit:

1. `node --check` the extracted `<script>` block.
2. Headless-run the data logic in `node` against `explorer/points.json`.
3. Cross-check any statistic against scipy in Python (e.g. the in-browser
   Spearman matches `scipy.stats.spearmanr` exactly).
4. Ask the user to eyeball it in a real browser (both themes) before it ships.

## Deploy

Push to `main`; GitHub Pages rebuilds from the repo root. The Fastly edge caches
~10 min (`max-age=600`), so a pushed change can take a few minutes to appear —
poll the live URL for the new markup rather than assuming it's instant.

## House rules

- Commit messages end with the Co-Authored-By trailer already used in history.
- The 214 MB of per-block activations in `results/clip_layers/` are tracked on
  purpose (rerun without a GPU); the unzipped crop/embedding dirs are not.
