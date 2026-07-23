# notebooks/

## `instance_segmentation.ipynb`

A fully-documented, runnable walkthrough of how we group the BabyView object
crops into **individual object instances** — the analysis behind the "Object
instances" panel of the
[explorer](https://vislearnlab.github.io/babyview-variability-explorer/) and the
`instance_decomposition` figure.

It is written to be **read top to bottom**: every code cell is short and preceded
by a plain-language explanation of what it does and why. The narrative arc:

1. **Load** the public crops + DINOv3 embeddings (de-normalize the stored z-score).
2. **Why DINOv3, not CLIP** — a histogram showing CLIP saturates and can't
   separate instances while DINOv3 can.
3. **The method** — average-linkage agglomerative clustering on cosine distance,
   with the threshold τ calibrated by contrast (clock/oven should repeat;
   book/car should not).
4. **Run + validate** — cluster all 72 categories and render montages (rows =
   estimated instances) so the grouping is checked *by eye*.
5. **The sampling confound, taken seriously** — separates the "duplicate frames"
   worry (testable, and largely ruled out) from the "object persistence" worry
   (not testable on public data; needs frame/session metadata).
6. **Payoff** — instance diversity indexes **local** dispersion (ρ ≈ +0.30, sig)
   but **not global** (≈ 0), read through the §5 caveat.

### Run it

From the repo root, with the public embeddings unzipped
(`cd data/valid7018_public && unzip -o embeddings.zip -d embeddings && unzip -o crops.zip -d crops`):

```bash
# open interactively
jupyter lab notebooks/instance_segmentation.ipynb

# or execute headless (embeds all outputs/figures in place)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=600 notebooks/instance_segmentation.ipynb
```

Pure CPU, no GPU or deep-learning framework needed at analysis time — the DINOv3
embeddings are precomputed `.npy` files. Runs in well under a minute.

### Regenerate the notebook itself

The notebook is assembled programmatically (so markdown + code stay readable and
diffable in Python rather than raw `.ipynb` JSON):

```bash
python scripts/build_instance_notebook.py     # writes the .ipynb
# then execute it as above to embed outputs
```

Edit `scripts/build_instance_notebook.py`, not the `.ipynb`, and re-run.

### Relationship to the scripts

`scripts/cluster_instances.py` is the non-notebook version of the same pipeline
— it writes `results/instances/instance_counts.csv`,
`instance_dispersion.csv`, and the montage JPEGs. The notebook is the
pedagogical companion; the script is what the explorer's data build reuses.

### Caveats (also stated in the notebook)

- **No ground truth** and the threshold τ is a transparent judgement call.
- **Sampling confound:** instance repetition is entangled with how crops were
  sampled from frames (a stationary object is seen in more frames). Duplicate
  frames are largely ruled out (§5), but persistence-driven over-sampling is
  not, and cannot be from the public data. Treat the instance→local-dispersion
  link as *"local dispersion partly reflects object persistence/repetition in
  the sample,"* not a clean count of distinct exemplars.
- 72 public categories only (body parts withheld); n = 72 is modest.
