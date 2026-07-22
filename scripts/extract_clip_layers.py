#!/usr/bin/env python3
"""Extract per-layer CLIP ViT-B/32 activations for the valid7018 public crops.

The CCN 2026 release ships only the final 512-d projected CLIP embedding. This
script re-runs the same model (OpenAI ViT-B/32, quickgelu) over the public crop
JPEGs and saves a readout from every one of the 12 transformer blocks, so the
paper's dispersion metrics can be recomputed as a function of depth.

Readouts per block L (0-indexed, 0 = first block, 11 = last):
  cls[L]   -- CLS token at the output of block L                     (768-d)
  mean[L]  -- mean over the 49 patch tokens at the output of block L (768-d)
Plus the model's own final readout:
  final    -- ln_post(CLS_11) @ proj, i.e. encode_image()            (512-d)

`final` is what the paper used. Sanity check in verify_against_release.py
confirms it matches the released vectors (r ~= .985; the gap is JPEG
re-encoding in the public crop archive, not a model mismatch).

Output (results/clip_layers/):
  manifest.csv        category, stem, row  -- row indexes every matrix below
  cls_L{00..11}.npy   (N, 768) float16
  mean_L{00..11}.npy  (N, 768) float16
  final.npy           (N, 512) float16
  extract_run.json    provenance

Run from repo root::

  python scripts/extract_clip_layers.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import open_clip

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "data" / "valid7018_public"
OUT = REPO / "results" / "clip_layers"

ARCH = "ViT-B-32-quickgelu"
PRETRAINED = "openai"
N_BLOCKS = 12


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_manifest() -> pd.DataFrame:
    """Crop rows in embedding-manifest order, so `row` matches the release."""
    emb = pd.read_csv(PUBLIC / "embeddings" / "manifest.csv")
    crops = pd.read_csv(PUBLIC / "crops" / "manifest.csv")
    df = emb.merge(crops, on=["category", "stem"], how="inner", validate="1:1")
    if len(df) != len(emb):
        raise SystemExit(f"crop/embedding manifests disagree: {len(df)} vs {len(emb)}")
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N crops")
    args = ap.parse_args()

    device = pick_device(args.device)
    df = build_manifest()
    if args.limit:
        df = df.head(args.limit).copy()
    n = len(df)
    print(f"{n} crops, {df.category.nunique()} categories, device={device}")

    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    model.eval().to(device)
    visual = model.visual
    blocks = visual.transformer.resblocks
    if len(blocks) != N_BLOCKS:
        raise SystemExit(f"expected {N_BLOCKS} blocks, found {len(blocks)}")

    # Hook every block; open_clip 3.x runs the visual tower batch-first, but we
    # detect the layout from the shape rather than trusting the flag.
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            captured[idx] = out.detach()

        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]

    cls_out = np.zeros((N_BLOCKS, n, 768), dtype=np.float16)
    mean_out = np.zeros((N_BLOCKS, n, 768), dtype=np.float16)
    final_out = np.zeros((n, 512), dtype=np.float16)

    paths = [PUBLIC / "crops" / p for p in df.jpeg_path]
    bs = args.batch_size
    for start in range(0, n, bs):
        chunk = paths[start : start + bs]
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        b = batch.shape[0]
        captured.clear()
        with torch.no_grad():
            feats = model.encode_image(batch)
        final_out[start : start + b] = feats.float().cpu().numpy().astype(np.float16)

        for layer in range(N_BLOCKS):
            tok = captured[layer]
            if tok.shape[0] == b and tok.shape[1] != b:
                pass  # (B, seq, width)
            elif tok.shape[1] == b:
                tok = tok.transpose(0, 1)  # (seq, B, width) -> (B, seq, width)
            else:
                raise SystemExit(f"ambiguous token layout {tuple(tok.shape)} for batch {b}")
            tok = tok.float().cpu().numpy()
            cls_out[layer, start : start + b] = tok[:, 0, :].astype(np.float16)
            mean_out[layer, start : start + b] = tok[:, 1:, :].mean(axis=1).astype(np.float16)

        if (start // bs) % 10 == 0:
            print(f"  {min(start + b, n)}/{n}", flush=True)

    for h in handles:
        h.remove()

    OUT.mkdir(parents=True, exist_ok=True)
    df[["category", "stem", "row"]].to_csv(OUT / "manifest.csv", index=False)
    for layer in range(N_BLOCKS):
        np.save(OUT / f"cls_L{layer:02d}.npy", cls_out[layer])
        np.save(OUT / f"mean_L{layer:02d}.npy", mean_out[layer])
    np.save(OUT / "final.npy", final_out)

    (OUT / "extract_run.json").write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "arch": ARCH,
                "pretrained": PRETRAINED,
                "n_blocks": N_BLOCKS,
                "n_exemplars": int(n),
                "n_categories": int(df.category.nunique()),
                "device": str(device),
                "readouts": {
                    "cls_L{i}": "CLS token at output of block i (768-d)",
                    "mean_L{i}": "mean of 49 patch tokens at output of block i (768-d)",
                    "final": "ln_post(CLS) @ proj -- encode_image(), as used in the CCN paper (512-d)",
                },
                "source_crops": "data/valid7018_public/crops (public privacy-filtered release)",
                "dtype": "float16",
                "note": "Raw activations, NOT z-scored. Normalization is applied per-analysis.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
