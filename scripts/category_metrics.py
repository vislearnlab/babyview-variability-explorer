"""Category dispersion metrics, copied verbatim from the CCN 2026 pipeline.

Source: object-detection @ e4883c6
        analysis/ccn-2026/scripts/valid7018_category_metrics.py

Kept byte-for-byte (modulo this docstring) so the layer-wise numbers are
computed by exactly the same code as the published final-layer numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def compute_category_metrics(
    category_embeddings: dict[str, np.ndarray], k: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for cat in sorted(category_embeddings.keys()):
        x = category_embeddings[cat]
        n = x.shape[0]
        effective_k = min(k, n - 1)
        if effective_k < 1:
            rows.append(
                {
                    "category": cat,
                    "n_exemplars": n,
                    "k": k,
                    "effective_k": effective_k,
                    "mean_knn_dist": np.nan,
                    "mean_pairwise_dist": np.nan,
                    "local_coherence": np.nan,
                    "global_dispersion": np.nan,
                    "local_over_global": np.nan,
                }
            )
            continue

        nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        nn.fit(x)
        dists, _ = nn.kneighbors(x)
        mean_knn = np.mean(dists[:, 1:], axis=1)
        pw = pairwise_distances(x, metric="euclidean")
        iu = np.triu_indices(n, k=1)
        mean_pairwise_dist = float(np.mean(pw[iu])) if len(iu[0]) > 0 else np.nan
        centroid = np.mean(x, axis=0, keepdims=True)
        centroid_dist = np.linalg.norm(x - centroid, axis=1)

        mean_knn_dist = float(np.mean(mean_knn))
        local_coherence = float(1.0 / mean_knn_dist) if mean_knn_dist > 0 else np.nan
        global_dispersion = float(np.mean(centroid_dist))
        local_over_global = (
            float(local_coherence / global_dispersion) if global_dispersion > 0 else np.nan
        )
        rows.append(
            {
                "category": cat,
                "n_exemplars": n,
                "k": k,
                "effective_k": effective_k,
                "mean_knn_dist": mean_knn_dist,
                "mean_pairwise_dist": mean_pairwise_dist,
                "local_coherence": local_coherence,
                "global_dispersion": global_dispersion,
                "local_over_global": local_over_global,
            }
        )
    return pd.DataFrame(rows).sort_values("local_over_global", ascending=False).reset_index(
        drop=True
    )
