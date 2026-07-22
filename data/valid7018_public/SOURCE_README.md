# valid7018 public release (privacy-filtered)

Paired crop JPEGs + CLIP/DINOv3 embeddings with sensitive categories removed.

## Exclusions

Body parts (15 labels): ankle, arm, ear, eye, face, finger, foot, hair, hand, leg, mouth, neck, nose, toe, tooth

Face privacy: glasses

Categories present in the source cohort that were dropped:
arm, ear, eye, face, finger, foot, glasses, hair, hand, mouth, nose, toe, tooth

## Contents

- `crops/valid7018_public_crops.zip` — 5,921 JPEGs, 72 categories
- `embeddings/valid7018_public_embeddings.zip` — matching CLIP + DINOv3 `.npy`
- `embeddings/valid7018_embedding_norm_stats.json` — cohort z-score μ/σ (fit on full 7,018)
- `valid7018_public_release.json` — build metadata

Stems are opaque ids (no subject/file identifiers). Manifests align crops ↔ embeddings 1:1.

Rebuild::

  python analysis/ccn-2026/scripts/build_valid7018_public_release.py
