# CLIP layer-wise analyses of BabyView object-category variability

Exploratory follow-up to the CCN 2026 extended abstract *"Variability in Young
Children's Everyday Visual Experiences of Object Categories"* (Yang, Sepuri,
Tan, Aw, Frank, & Long).

The abstract measured category variability using the **final** CLIP ViT-B/32
embedding. This repo asks where in the network that variability structure
actually arises — recomputing the same dispersion metrics at **every one of the
12 transformer blocks**.

Motivating limitation, from the abstract's Discussion:

> "our metrics do not distinguish between different sources of exemplar
> variability; developing new quantitative metrics and relating these to
> learning outcomes remains an exciting avenue for future work."

Depth is one handle on that: if a category's dispersion ranking is already
present in block 0, it is largely low-level appearance (color, texture,
lighting); if it only appears in the last blocks, it is a semantic/viewpoint
property that the network constructs.

## Data

Everything comes from the **public, privacy-filtered** release in
[`babyview-project/object-detection`](https://github.com/babyview-project/object-detection)
at commit [`e4883c6`](https://github.com/babyview-project/object-detection/commit/e4883c61752235addf3e6572998213b519c0be70)
(`data/shared_data_ccn_2026/public/`):

- **5,921 crops, 72 categories** (the paper's 85 minus 13 body-part/glasses
  categories withheld for privacy)
- released final-layer CLIP (512-d) + DINOv3 (768-d) vectors, feature-wise
  z-scored on the full 7,018-crop cohort
- the published `valid7018` per-category metric CSVs, for comparison

No non-public BabyView data is used or required.

## Pipeline

```
python scripts/extract_clip_layers.py     # 12 blocks x 5,921 crops  (~4 min, MPS)
python scripts/compute_layer_metrics.py   # dispersion metrics per readout
python scripts/plot_layer_curves.py       # figures
```

`scripts/category_metrics.py` is a **verbatim copy** of the paper's
`analysis/ccn-2026/scripts/valid7018_category_metrics.py`, so the layer-wise
numbers come from exactly the same metric code as the published ones.

**Model.** OpenAI `ViT-B-32-quickgelu` via `open_clip`. Verified as the correct
checkpoint: re-extracting the final layer correlates **r = .985** per-image with
the released vectors (residual is JPEG re-encoding in the public crop archive);
LAION-2B weights correlate ~.00.

**Readouts per block L.** `cls_L{L}` (CLS token, 768-d) and `mean_L{L}` (mean of
the 49 patch tokens, 768-d), plus `final` (`ln_post(CLS) @ proj`, 512-d) — the
paper's readout. Each is feature-wise z-scored on the pooled 5,921-crop cohort
before distances, matching the paper's normalization form.

**Comparability caveat.** Euclidean distance in *d* z-scored dimensions scales
like sqrt(*d*), and blocks are 768-d while `final` is 512-d. Absolute dispersion
is therefore *not* comparable across readouts; all inference here rests on
**across-category rank correlations**.

## Findings

### 1. Pipeline validates against the published numbers

`final` reproduces the published per-category CLIP metrics at
**rho = .993** (global) and **rho = .994** (local) across the 72 categories.
Cross-model agreement on this cohort is rho = .411 global / .518 local,
versus the paper's .545 / .626 on 85 categories — the expected attenuation from
dropping the body-part categories, which sat at the low end of the range.

### 2. Global dispersion is a *late* property; local dispersion is not

Rank agreement with the published final-layer CLIP ordering (CLS path):

| block | 0 | 3 | 5 | 8 | 10 | 11 | final |
|---|---|---|---|---|---|---|---|
| **global** | .29 | .54 | .55 | .04 | .71 | .93 | .99 |
| **local (kNN)** | .65 | .56 | .52 | .29 | .83 | .95 | .99 |

The global (distance-to-centroid) ordering that Figure 1C of the abstract is
built on is **essentially absent until the last two blocks** — it is constructed
at blocks 10–11, not inherited from low-level image statistics. The single
largest reorganization is block 9 → 10 (adjacent-block rho = .60, against
.81–.95 everywhere else).

Local (kNN) dispersion behaves differently: it already agrees at **rho = .65 in
block 0**. Which exemplars sit near each other is partly a low-level fact; how
far the category as a whole spreads is not.

That dissociation is the substantive result — the paper's two metrics, which
correlate well at the final layer, have genuinely different origins in depth.

### 3. The weak frequency–dispersion link is not a readout artifact

The abstract reports frequency vs. global dispersion at rho = .18 (CLIP, n.s.)
and .26 (DINOv3). A natural worry is that the final projection is simply the
wrong place to look. It is not: **no readout at any depth shows a reliable
coupling** (all |rho| < .24; the largest, `mean_L03` at rho = .23, p = .05, does
not survive 25 comparisons). Frequency and variability look genuinely
dissociable across the whole network, which strengthens the paper's claim.

### 4. CLIP's patch tokens align with DINOv3 better than CLIP's CLS does

For local dispersion, `mean_L08`–`mean_L11` agree with published DINOv3 at
rho = .60–.62, exceeding the final CLIP readout's own agreement with DINOv3
(.52). Plausibly because DINOv3's training objective is patch-level: the
cross-model agreement the abstract reports is likely an underestimate of how
much the two models actually share, and is partly a readout mismatch rather than
a representational disagreement.

## Incidental finding: a stale export in `data/shared_data_ccn_2026/valid7018/`

The `valid7018` metric CSVs exist in two places in the upstream repo, and they
disagree. The `analysis/ccn-2026/valid7018/` copies are correct and match the
abstract exactly; the `data/shared_data_ccn_2026/valid7018/` copies are frozen
at an older normalization.

| | shared bundle | `analysis/` copy | abstract |
|---|---|---|---|
| CLIP global mean (85 cats) | 24.46 | **18.12** | **18.12** |
| CLIP global max (`book`) | 29.17 | **21.59** | **21.59** |
| DINOv3 global mean | 32.08 | **23.36** | **23.36** |
| DINOv3 global max (`car`) | 38.28 | **27.81** | **27.81** |

Cause, from the history: `1c504b4` (Jun 8) last wrote the shared bundle;
`f535af9` immediately after ("Switch valid7018 normalization to cohort-internal
feature-wise z-score") regenerated `analysis/ccn-2026/valid7018/`, the norm-stats
JSON, `valid7018_paper_stats.json`, and the figures — but never re-exported to
`data/shared_data_ccn_2026/`. 7 of 8 files in the shared directory are stale.

Recomputing from the vectors shipped in `shared_data_ccn_2026/public/` with the
repo's own metric code reproduces the `analysis/` values to 5 decimals
(ratio 1.00000), so the shared *embeddings* are current — only the CSVs beside
them are not. The old normalization differs by a near-constant scale factor
(1.351 CLIP, 1.376 DINOv3), so ranks are untouched (r = .9999) and no conclusion
in the abstract is affected. See [NOTE_FOR_JANE.md](NOTE_FOR_JANE.md).

## Layout

```
data/valid7018_public/    public crops + released embeddings (from e4883c6)
data/ccn2026_results/     published metric CSVs, for comparison
scripts/                  extraction, metrics, plotting
results/clip_layers/      per-block activations (214 MB, gitignored)
results/metrics/          layer_category_metrics.csv, baseline_..., layer_summary.csv
results/figures/          fig1_emergence, fig2_frequency, poster_layerwise
explorer/                 interactive page: index.html + points.json + crops/
```

## Explorer

`explorer/` is an interactive page (same house style as the vislearnlab drawing
explorers) with a **layer slider**: 5,921 crops laid out by t-SNE at each CLIP
block, so you can watch the space reorganize from block 0 to the final embedding.
Color by CDI semantic group or by dispersion, filter by category, hover for the
crop. See [explorer/README.md](explorer/README.md).

```
cd explorer && python3 -m http.server 8000
```

Intended for GitHub Pages, so it can go on the poster as a QR link.

## Status

Exploratory follow-up, not yet reviewed by co-authors.
