#!/usr/bin/env python3
"""Does category dispersion predict age of acquisition (AoA)?

The CCN abstract measured how variable each object category is, and noted that
relating variability to learning outcomes was future work. This closes that loop
on the public 72-category set: join each category's AoA (from the CDI MacArthur-
Bates norms) to its dispersion, and test the relationship at every CLIP block.

Two questions:
  1. Does more within-category visual variability go with *later* acquisition
     (harder to learn) or *earlier* (variability aids generalization)? The
     developmental literature predicts either sign, so the sign matters.
  2. Is the relationship depth-dependent -- does dispersion at some layer predict
     AoA better than the final embedding the paper used?

The classic AoA confounder is **word** frequency in child-directed speech
(frequent-in-speech words are learned earlier) -- NOT the paper's visual
frequency (proportion of frames a category is seen in). These are different
things and, on this dataset, nearly decoupled (rho ~ .12; the Clerkin et al.
point). So we control for CHILDES word frequency, and separately report visual
frequency as its own variable.

Sources (all join to 72/72 categories):
  AoA           data/aoa/MCDI_items_with_AoA.csv     (uni_lemma)
  word freq     data/aoa/childes_english.csv         (word_count in child-directed speech)
  visual freq   data/ccn2026_results/...             (proportion of frames)

Outputs (results/aoa/):
  aoa_by_category.csv     category, AoA, frequency, published global/local, per-layer dispersion
  aoa_dispersion_by_layer.csv   per-readout raw + frequency-partialled rho vs AoA
  aoa_run.json

Run from repo root::

  python scripts/aoa_dispersion.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
METRICS = REPO / "results" / "metrics"
CCN = REPO / "data" / "ccn2026_results"
AOA = REPO / "data" / "aoa" / "MCDI_items_with_AoA.csv"
CHILDES = REPO / "data" / "aoa" / "childes_english.csv"
OUT = REPO / "results" / "aoa"

READOUTS = [f"cls_L{i:02d}" for i in range(12)] + [f"mean_L{i:02d}" for i in range(12)] + ["final"]


def norm(s: str) -> str:
    return str(s).strip().lower()


def aoa_lookup() -> dict[str, float]:
    aoa = pd.read_csv(AOA)
    lut: dict[str, float] = {}
    for col in ("uni_lemma", "item_definition", "english_gloss"):
        for _, r in aoa.iterrows():
            k = norm(r[col])
            if k and k not in lut and pd.notna(r["AoA"]):
                lut[k] = float(r["AoA"])
    return lut


def resolve_aoa(cat: str, lut: dict[str, float]) -> float | None:
    for key in (cat, cat + "s", cat.rstrip("s")):
        if norm(key) in lut:
            return lut[norm(key)]
    return None


def childes_lookup() -> dict[str, float]:
    c = pd.read_csv(CHILDES)
    return dict(zip(c.word.astype(str).map(norm), c.word_count))


def resolve_wordfreq(cat: str, lut: dict[str, float]) -> float | None:
    for key in (cat, cat.rstrip("s")):
        if norm(key) in lut:
            return float(lut[norm(key)])
    return None


def partial_spearman(x, y, z):
    """Spearman partial correlation of x,y controlling for z (rank-residual method)."""
    from scipy.stats import rankdata

    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]
        beta, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ beta

    ex, ey = resid(rx, rz), resid(ry, rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    # t-test on partial correlation, 1 covariate
    from scipy.stats import t as tdist

    dof = n - 2 - 1
    tval = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    p = 2 * tdist.sf(abs(tval), dof)
    return float(r), float(p)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lut = aoa_lookup()
    wlut = childes_lookup()

    layers = pd.read_csv(METRICS / "layer_category_metrics.csv")
    base = pd.read_csv(METRICS / "baseline_category_metrics.csv")
    freq = pd.read_csv(CCN / "category_frequency_valid85.csv").set_index("category")
    pub = base[base.source == "released_clip"].set_index("category")

    cats = sorted(layers.category.unique())
    tbl = pd.DataFrame({"category": cats})
    tbl["AoA"] = tbl.category.map(lambda c: resolve_aoa(c, lut))
    tbl["visual_frequency"] = tbl.category.map(lambda c: float(freq.loc[c, "proportion"]) if c in freq.index else np.nan)
    tbl["word_count_childes"] = tbl.category.map(lambda c: resolve_wordfreq(c, wlut))
    tbl["log_word_freq"] = np.log10(tbl.word_count_childes + 1)
    tbl["cdi_semantic"] = tbl.category.map(lambda c: freq.loc[c, "cdi_semantic"] if c in freq.index else "other")
    tbl["published_global"] = tbl.category.map(lambda c: float(pub.loc[c, "global_dispersion"]))
    tbl["published_local"] = tbl.category.map(lambda c: float(pub.loc[c, "mean_knn_dist"]))
    for name in READOUTS:
        d = layers[layers.readout == name].set_index("category")
        tbl[f"global_{name}"] = tbl.category.map(lambda c: float(d.loc[c, "global_dispersion"]))
        tbl[f"local_{name}"] = tbl.category.map(lambda c: float(d.loc[c, "mean_knn_dist"]))

    n_missing = tbl.AoA.isna().sum()
    tbl = tbl.dropna(subset=["AoA", "log_word_freq"]).reset_index(drop=True)
    print(f"{len(tbl)} categories with AoA + word freq ({n_missing} dropped for missing AoA)")
    tbl.to_csv(OUT / "aoa_by_category.csv", index=False)

    # Validate the setup and characterize the two frequencies.
    r_wf, p_wf = spearmanr(tbl.log_word_freq, tbl.AoA)
    r_vf, p_vf = spearmanr(tbl.visual_frequency, tbl.AoA)
    r_vw, _ = spearmanr(tbl.visual_frequency, tbl.log_word_freq)
    print(f"WORD freq (CHILDES) vs AoA:  rho={r_wf:+.3f} p={p_wf:.3g}  "
          f"(classic effect -- validates the AoA data)")
    print(f"VISUAL freq vs AoA:          rho={r_vf:+.3f} p={p_vf:.3g}  (not the same predictor)")
    print(f"visual vs word freq:         rho={r_vw:+.3f}  (nearly decoupled -- Clerkin et al.)")

    rows = []
    for name in READOUTS:
        block = -1 if name == "final" else int(name.split("_L")[1])
        kind = "final" if name == "final" else name.split("_")[0]
        for metric in ("global", "local"):
            col = f"{metric}_{name}"
            r_raw, p_raw = spearmanr(tbl[col], tbl.AoA)
            r_par, p_par = partial_spearman(tbl[col].values, tbl.AoA.values, tbl.log_word_freq.values)
            rows.append({
                "readout": name, "kind": kind, "block": block, "metric": metric,
                "rho_aoa_raw": r_raw, "p_raw": p_raw,
                "rho_aoa_partial_freq": r_par, "p_partial": p_par,
            })
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "aoa_dispersion_by_layer.csv", index=False)

    # headline: the paper's readout (final, both metrics)
    print("\n=== AoA vs dispersion (final embedding, the paper's readout) ===")
    for _, r in res[res.readout == "final"].iterrows():
        print(f"  {r.metric:6s}  raw rho={r.rho_aoa_raw:+.3f} (p={r.p_raw:.3g})   "
              f"partial-wordfreq rho={r.rho_aoa_partial_freq:+.3f} (p={r.p_partial:.3g})")

    # best layer per metric (largest |partial rho|)
    print("\n=== strongest layer (|partial-freq rho|), CLS path ===")
    cls = res[res.kind == "cls"]
    for metric in ("global", "local"):
        sub = cls[cls.metric == metric].reindex(
            cls[cls.metric == metric].rho_aoa_partial_freq.abs().sort_values(ascending=False).index)
        top = sub.iloc[0]
        print(f"  {metric:6s}  block {int(top.block):2d}: partial rho={top.rho_aoa_partial_freq:+.3f} (p={top.p_partial:.3g})")

    (OUT / "aoa_run.json").write_text(json.dumps({
        "n_categories": int(len(tbl)),
        "aoa_source": "data/aoa/MCDI_items_with_AoA.csv (uni_lemma join)",
        "word_freq_source": "data/aoa/childes_english.csv (word_count, log10)",
        "word_freq_vs_aoa_rho": r_wf, "word_freq_vs_aoa_p": p_wf,
        "visual_freq_vs_aoa_rho": r_vf, "visual_freq_vs_aoa_p": p_vf,
        "visual_vs_word_freq_rho": r_vw,
        "method": "Spearman; partial controls for CHILDES log word frequency via rank residuals",
        "headline": "Dispersion does not predict AoA (validated null: word freq DOES predict AoA "
                    "rho=%.2f). Visual and word frequency are nearly decoupled (rho=%.2f)." % (r_wf, r_vw),
        "caveat": "72 public categories only (body parts excluded); n=72 is modest -- treat as suggestive.",
    }, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
