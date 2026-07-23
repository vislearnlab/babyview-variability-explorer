#!/usr/bin/env python3
"""Extract per-layer DINOv3 ViT-B/16 activations for the valid7018 public crops.

Companion to extract_clip_layers.py. The paper reports both CLIP and DINOv3 but
this repo's layer analysis was CLIP-only; this adds DINOv3 so the "global built
late / local early" result can be tested across both models. DINOv3 is
self-supervised (no language), so agreement would say the depth pattern is about
vision hierarchies in general, not CLIP's text alignment.

Model: facebook/dinov3-vitb16-pretrain-lvd1689m -- the checkpoint named in the
paper's image-embedding/create_image_embeddings.py. Gated on HF; needs a token
(`huggingface-cli login`).

Readouts per block L (0-indexed, matching the CLIP script):
  cls[L]   -- CLS token at the output of block L                  (768-d)
  mean[L]  -- mean over PATCH tokens at the output of block L      (768-d)
             (excludes CLS *and* the register tokens; DINOv3 puts
              num_register_tokens registers between CLS and patches)
Plus the model's own final readout:
  final    -- pooler_output = CLS after the backbone's final LayerNorm (768-d),
              which is the readout the paper used for DINOv3.

Sanity: verify_against_release equivalent is inline -- `final` must match the
released DINOv3 vectors (they were built from this same pooler_output). We print
the per-image correlation before writing anything.

Output (results/dinov3_layers/): same shape as results/clip_layers/.

Run from repo root (after HF login)::

  python scripts/extract_dinov3_layers.py
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

# DINOv3's image processor is fast-only and calls torch.compiler.is_compiling(),
# which does not exist in torch 2.2.x. Shim it to False (normal eager path) so
# the processor runs unchanged; is_rocm_platform() is then short-circuited too.
import types as _types
if not hasattr(torch, "compiler"):
    torch.compiler = _types.SimpleNamespace()
if not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

from transformers import AutoImageProcessor, AutoModel

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "data" / "valid7018_public"
OUT = REPO / "results" / "dinov3_layers"

MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
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


def released_dinov3(df: pd.DataFrame) -> np.ndarray:
    """Released DINOv3 vectors, de-normalized to raw space, for the sanity check."""
    st = json.loads((PUBLIC / "embedding_norm_stats.json").read_text())["models"]["dinov3"]
    mu, sd = np.array(st["mu"]), np.array(st["sigma"])
    return np.stack([np.load(PUBLIC / "embeddings" / p).astype(np.float32) * sd + mu
                     for p in df.dinov3_npy])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-only", action="store_true", help="only run the reproduce check, then stop")
    args = ap.parse_args()

    device = pick_device(args.device)
    df = build_manifest()
    if args.limit:
        df = df.head(args.limit).copy()
    n = len(df)
    print(f"{n} crops, {df.category.nunique()} categories, device={device}")

    processor = AutoImageProcessor.from_pretrained(MODEL)  # fast-only; shimmed above
    model = AutoModel.from_pretrained(MODEL).eval().to(device)
    n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    hidden = int(model.config.hidden_size)
    n_layers = int(model.config.num_hidden_layers)
    print(f"DINOv3: {n_layers} blocks, hidden={hidden}, register_tokens={n_reg}")
    if n_layers != N_BLOCKS:
        raise SystemExit(f"expected {N_BLOCKS} blocks, found {n_layers}")

    paths = [PUBLIC / "crops" / p for p in df.jpeg_path]
    cls_out = np.zeros((N_BLOCKS, n, hidden), dtype=np.float16)
    mean_out = np.zeros((N_BLOCKS, n, hidden), dtype=np.float16)
    final_out = np.zeros((n, hidden), dtype=np.float32)  # keep fp32 for the check

    bs = args.batch_size
    patch0 = 1 + n_reg  # first real patch token index (after CLS + registers)
    for start in range(0, n, bs):
        chunk = paths[start:start + bs]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        px = processor(images=imgs, return_tensors="pt").to(device)
        b = len(chunk)
        with torch.no_grad():
            out = model(**px, output_hidden_states=True)
        # hidden_states: (n_layers+1) tensors of (B, seq, hidden); [0]=embeddings,
        # [i]=output of block i-1. Block L output = hidden_states[L+1].
        hs = out.hidden_states
        final_out[start:start + b] = out.pooler_output.float().cpu().numpy()
        for layer in range(N_BLOCKS):
            tok = hs[layer + 1].float().cpu().numpy()  # (b, seq, hidden)
            cls_out[layer, start:start + b] = tok[:, 0, :].astype(np.float16)
            mean_out[layer, start:start + b] = tok[:, patch0:, :].mean(axis=1).astype(np.float16)
        if (start // bs) % 10 == 0:
            print(f"  {min(start + b, n)}/{n}", flush=True)

    # ---- sanity: `final` must reproduce the released DINOv3 vectors ----------
    rel = released_dinov3(df)
    cs = [np.corrcoef(final_out[i], rel[i])[0, 1] for i in range(min(n, 200))]
    mean_r = float(np.mean(cs))
    print(f"\nreproduce released DINOv3 (pooler_output vs release): "
          f"mean r={mean_r:.4f} min={np.min(cs):.4f}  (first {len(cs)} crops)")
    if mean_r < 0.9:
        print("!! WARNING: final readout does not match the release (r<0.9). "
              "Check the model/processor before trusting the block readouts.")
    if args.check_only:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    df[["category", "stem", "row"]].to_csv(OUT / "manifest.csv", index=False)
    for layer in range(N_BLOCKS):
        np.save(OUT / f"cls_L{layer:02d}.npy", cls_out[layer])
        np.save(OUT / f"mean_L{layer:02d}.npy", mean_out[layer])
    np.save(OUT / "final.npy", final_out.astype(np.float16))

    (OUT / "extract_run.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "n_blocks": N_BLOCKS,
        "hidden_size": hidden,
        "num_register_tokens": n_reg,
        "n_exemplars": int(n),
        "n_categories": int(df.category.nunique()),
        "device": str(device),
        "reproduce_released_final_mean_r": mean_r,
        "readouts": {
            "cls_L{i}": "CLS token at output of block i (768-d)",
            "mean_L{i}": "mean of patch tokens (excl. CLS + registers) at output of block i (768-d)",
            "final": "pooler_output = CLS after final LayerNorm -- the paper's DINOv3 readout (768-d)",
        },
        "source_crops": "data/valid7018_public/crops (public privacy-filtered release)",
        "dtype": "float16",
        "note": "Raw activations, NOT z-scored. Normalization applied per-analysis, matching CLIP.",
    }, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
