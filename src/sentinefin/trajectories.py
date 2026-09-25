"""Phase 4: trajectory construction from DEC outputs.

Bridges Phase 3 (per-window DEC clusters) and Phase 4 (LSTM-AE scoring):
builds one feature time series per global cluster across consecutive windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class TrajectoryBundle:
    """Cluster trajectories plus bookkeeping for reporting."""

    sequences: list[np.ndarray] = field(default_factory=list)
    meta: list[dict] = field(default_factory=list)


def build_trajectories_from_frames(
    panel: pd.DataFrame,
    assignments: pd.DataFrame,
    vectors: np.ndarray,
    centroids_by_window: dict[str, np.ndarray],
    local_to_global_by_window: dict[str, list[int]],
) -> TrajectoryBundle:
    """One feature time series per global cluster.

    Per (cluster, window) features (in FEATURE_NAMES order):
    - centroid_shift: latent movement of the cluster centroid since the
      cluster's previous window
    - volume: log1p of complaints assigned to the cluster
    - volume_share: fraction of the window's complaints in this cluster
    - mean_dist: mean distance of member embeddings to their centroid
    """
    merged = assignments.merge(panel[["complaint_id", "window_id"]], on="complaint_id", how="left")
    # Compress raw embedding scale so mean_dist stays comparable to other features.
    latents = vectors / max(1.0, float(np.sqrt(vectors.shape[1])))
    id_to_pos = {cid: i for i, cid in enumerate(panel["complaint_id"].tolist())}

    series: dict[int, list[list[float]]] = {}
    first_window: dict[int, str] = {}
    prev_centroids: dict[int, np.ndarray] = {}

    for w in sorted(merged["window_id"].dropna().unique()):
        sub = merged[merged["window_id"] == w]
        if w not in centroids_by_window:
            continue
        cents_w = centroids_by_window[w]
        l2g = local_to_global_by_window[w]
        total_w = float(len(sub))
        for local_c in sorted(sub["local_cluster"].unique()):
            g = int(l2g[int(local_c)])
            pts = sub[sub["local_cluster"] == local_c]
            pos = np.array(
                [id_to_pos[cid] for cid in pts["complaint_id"] if cid in id_to_pos], dtype=int
            )
            volume = float(len(pts))
            if len(pos):
                vecs = latents[pos]
                mean_dist = float(
                    np.linalg.norm(vecs - cents_w[int(local_c)][None, :], axis=1).mean()
                )
            else:
                mean_dist = 0.0
            shift = (
                float(np.linalg.norm(cents_w[int(local_c)] - prev_centroids[g]))
                if g in prev_centroids
                else 0.0
            )
            prev_centroids[g] = cents_w[int(local_c)].copy()
            feat = [
                shift,
                float(np.log1p(volume)),
                volume / total_w,
                mean_dist,
            ]
            series.setdefault(g, []).append(feat)
            first_window.setdefault(g, w)

    bundle = TrajectoryBundle()
    for g in sorted(series.keys()):
        seq = np.asarray(series[g], dtype=np.float32)
        bundle.sequences.append(seq)
        bundle.meta.append(
            {
                "global_cluster": int(g),
                "first_window": first_window[g],
                "length": int(len(seq)),
            }
        )
    return bundle
