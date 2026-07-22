# Note for Jane — CLIP layer follow-up, plus one stale file in the shared bundle

Hi Jane — I did some exploratory follow-up on the CCN abstract, looking at where
in CLIP the variability structure actually comes from. Code and results are in
this repo (`object-detection-clip-layers`); everything runs off the public
release you built, so nothing here needs cluster access.

Two things: the results, and one small data-hygiene item.

---

## 1. Data hygiene: `data/shared_data_ccn_2026/valid7018/` never got re-exported

**Nothing is wrong with the paper.** I want to lead with that, because I went
looking at these files and briefly confused myself. Every number in the abstract
checks out exactly, and I verified it line by line — see the table below.

The only issue is that one *exported* directory in the repo was never refreshed
after the normalization switch, so it disagrees with everything else. No
analysis, figure, or embedding depends on it.

How it happened, from the history:

- `1c504b4` (Jun 8) — last commit that wrote `data/shared_data_ccn_2026/valid7018/`
- `f535af9` (Jun 8, immediately after) — "Switch valid7018 normalization to
  cohort-internal feature-wise z-score." This regenerated
  `analysis/ccn-2026/valid7018/`, the norm-stats JSON, `valid7018_paper_stats.json`,
  and the figures — but did not re-export to `data/shared_data_ccn_2026/`.

So the shared copy still reflects the older normalization:

| | shared bundle | `analysis/` copy | abstract |
|---|---|---|---|
| CLIP global, mean (85 cats) | 24.46 | **18.12** | **18.12** |
| CLIP global, max (`book`) | 29.17 | **21.59** | **21.59** |
| DINOv3 global, mean | 32.08 | **23.36** | **23.36** |
| DINOv3 global, max (`car`) | 38.28 | **27.81** | **27.81** |

### The paper is fine — every reported value verified

| abstract reports | current `analysis/` copy |
|---|---|
| CLIP global mean 18.12, SD 1.84, [13.10, 21.59] | 18.12, 1.84, [13.10, 21.59] ✓ |
| DINOv3 global mean 23.36, SD 1.76, [16.18, 27.81] | 23.36, 1.76, [16.18, 27.81] ✓ |
| DINOv3 local mean 27.05, SD 2.60, [17.49, 33.29] | 27.05, 2.60, [17.49, 33.29] ✓ |
| freq × dispersion, DINOv3 ρ = .26, p = .014 | ρ = .2654, p = .0141 ✓ |
| cross-model ρ = .55 global / .63 local | .551 / .623 ✓ |
| Fig 1A montages: clock, oven, chair, paper, book | `montage_low_to_high`, same order ✓ |
| Fig 1A V_c: 23.55, 23.21, 25.53, 22.98, 22.88 | DINOv3: 23.55, 23.21, 25.53, 22.98, 22.88 ✓ |

The stale values (24.46, 32.08) appear nowhere in the abstract. And the stale
copy's montage list is `[glasses, oven, balloon, paper, book]`, which is *not*
what Fig 1A shows — further confirmation the paper was built from the current
pipeline.

I also recomputed from the CLIP/DINOv3 vectors shipped in
`shared_data_ccn_2026/public/`, using your `valid7018_category_metrics.py`
unchanged, and got the `analysis/` values to 5 decimal places (ratio 1.00000).
So the *embeddings* in the shared bundle are current — it's only the CSVs
sitting next to them that are stale.

7 of the 8 files in `data/shared_data_ccn_2026/valid7018/` differ from their
`analysis/ccn-2026/valid7018/` counterparts (all but
`bv_valid7018_n_exemplars_by_category.csv`), including `valid7018_paper_stats.json`
and `valid7018_run.json`.

**Why it's worth fixing even though the paper is fine:** the shared bundle is
what we'd point an external person at, and it's internally inconsistent —
recomputing from the embeddings in that directory gives numbers ~1.35× different
from the CSVs in the same directory. That reads like a failed reproduction.

**Does it change any conclusion?** No. The old normalization differs by a
near-constant scale factor (1.351 CLIP, 1.376 DINOv3, constant to ~0.1% across
categories), so category rankings are untouched (r = .9999) and the correlations
barely move — cross-model global ρ = .545 stale vs .551 current, local
ρ = .626 vs .623. Both round to the .55/.63 in the abstract.

The fix is just re-running the export step (`build_shared_public_data_ccn.py`,
I think?) so the two directories agree.

---

## 2. The layer analyses

**Question.** The abstract measures dispersion in the final CLIP embedding. But
the Discussion notes we can't distinguish *sources* of exemplar variability.
Depth gives one handle on that: if a category's dispersion ranking is already
present in block 0, it's largely low-level appearance (color, texture, lighting);
if it only appears in the last blocks, it's something the network constructs.

**Setup.** Re-ran CLIP over the 5,921 public crops (72 categories) and pulled a
readout from all 12 transformer blocks — CLS token and mean-pooled patch tokens
— then recomputed global and local dispersion at each, using your
`valid7018_category_metrics.py` copied verbatim so the numbers come from the
same metric code.

Two sanity checks passed first: the checkpoint is OpenAI `ViT-B-32-quickgelu`
(re-extracting the final layer gives r = .985 per-image against the released
vectors; LAION-2B weights give ~.00 — worth recording somewhere, the repo
doesn't currently say which checkpoint was used), and my final-layer readout
reproduces the published per-category values at ρ = .993 global / .994 local.

### Main result: global and local dispersion have different origins in depth

Rank agreement with the published final-layer CLIP ordering, CLS path:

| block | 0 | 3 | 5 | 8 | 10 | 11 | final |
|---|---|---|---|---|---|---|---|
| **global** (dist. to centroid) | .29 | .54 | .55 | .04 | .71 | .93 | .99 |
| **local** (kNN, k=5) | .65 | .56 | .52 | .29 | .83 | .95 | .99 |

The global ordering that Fig 1C is built on is **essentially absent until the
last two blocks** — it's constructed at 10–11, not inherited from low-level
image statistics. Biggest single reorganization is block 9 → 10 (adjacent-block
ρ = .60, vs .81–.95 everywhere else).

Local dispersion behaves differently: already ρ = .65 at block 0. So *which
exemplars sit near each other* is partly a low-level fact, while *how far the
category spreads overall* is not.

That dissociation seems like the interesting bit — the two metrics agree well at
the final layer, which made them look somewhat redundant, but they have
genuinely different origins. It's at least a partial answer to the
"can't distinguish sources of variability" limitation, and it suggests the two
metrics might make different developmental predictions.

### The weak frequency–dispersion link is not a readout artifact

A reader could reasonably ask whether ρ = .18 / .26 is just the wrong readout.
It isn't — **no readout at any depth shows reliable coupling** (all |ρ| < .24;
largest is `mean_L03` at ρ = .23, p = .05, which doesn't survive 25 comparisons).
Frequency and variability look genuinely dissociable across the whole network.
I think this strengthens the claim and is worth a sentence if this goes to a
longer paper.

### CLIP's patch tokens match DINOv3 better than CLIP's CLS does

For local dispersion, `mean_L08`–`mean_L11` agree with DINOv3 at ρ = .60–.62,
above the final CLIP readout's own agreement with DINOv3 (.52 on these 72
categories). Plausibly because DINOv3's objective is patch-level. If so, the
cross-model agreement we report is partly a readout mismatch rather than a
representational disagreement — i.e. an underestimate of what the two models
actually share.

---

## Caveats

- **72 categories, not 85.** Public release excludes the 13 body-part/glasses
  categories. Those were the low-dispersion end, so the range is truncated —
  cross-model agreement on this cohort is ρ = .411 global / .518 local vs
  .551 / .623 on the full 85. All layer comparisons are internally consistent
  (same cohort throughout), but absolute agreement values aren't comparable to
  the abstract's.
- **No cross-layer magnitude claims.** Euclidean distance scales like √d and
  blocks are 768-d vs 512-d for the final projection, so everything rests on
  across-category rank correlations, not raw V_c.
- Layer z-scores are refit on the 5,921-crop public cohort (same normalization
  *form* as the paper, different cohort).
- Exploratory — no preregistration, and the mid-network dip at block 8 is based
  on n = 72 categories, so I wouldn't lean on its exact location.

## To reproduce

```
python scripts/extract_clip_layers.py     # ~4 min on an M3 Max
python scripts/compute_layer_metrics.py
python scripts/plot_layer_curves.py
```

Figures in `results/figures/`, per-category numbers in
`results/metrics/layer_category_metrics.csv`.

Happy to talk through any of this, especially whether the global/local
dissociation is worth chasing further.

— Bria
