"""End-to-end SentinelFin pipeline: ingest -> embed -> DEC -> LSTM-AE drift.

Every stage persists artifacts to disk so later stages (and the backtest) can
resume without recomputing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as _cfg
from .config import (
    DECConfig,
    DriftConfig,
    EmbeddingConfig,
    MLPConfig,
)
from .dec import run_dec_over_windows
from .drift import FEATURE_NAMES, choose_threshold, flag_alerts, score_trajectories, train_lstm_ae
from .embed import embed_panel
from .ingest import build_panel, load_raw, run_eda
from .mlp_baseline import train_mlp_baseline
from .trajectories import TrajectoryBundle, build_trajectories_from_frames
from .utils import save_json

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    panel: pd.DataFrame
    embeddings: np.ndarray
    assignments: pd.DataFrame
    dec_summary: dict
    trajectories: TrajectoryBundle
    drift_scores: dict
    alerts: list[dict]
    threshold: float


def run_pipeline(
    raw_path=None,
    smoke: bool = False,
    skip_eda: bool = False,
) -> PipelineResult:
    """Run Phases 1-4 end to end and persist every artifact."""
    from .config import DataConfig

    data_cfg = DataConfig()
    emb_cfg = EmbeddingConfig()
    mlp_cfg = MLPConfig()
    dec_cfg = DECConfig()
    drift_cfg = DriftConfig()

    # Phase 1 ------------------------------------------------------------
    if smoke:
        from .synthetic import write_synthetic_raw

        raw_path = write_synthetic_raw()
    else:
        raw_path = None
    raw = load_raw(raw_path) if raw_path is not None else load_raw()
    panel = build_panel(raw, config=data_cfg)
    logger.info("Panel: %d complaints across %d windows", len(panel), panel["window_id"].nunique())
    if not skip_eda:
        for p in run_eda(panel):
            logger.info("EDA plot: %s", p)

    # Phase 2 ------------------------------------------------------------
    vectors = embed_panel(panel, config=emb_cfg)
    try:
        metrics = train_mlp_baseline(vectors, panel["product"], config=mlp_cfg)
        logger.info("MLP baseline: %s", metrics)
    except Exception as exc:  # baseline must never block the pipeline
        logger.warning("MLP baseline skipped: %s", exc)

    # Phase 3 ------------------------------------------------------------
    assignments, dec_summary = run_dec_over_windows(panel, vectors, cfg=dec_cfg)

    # Reconstruct per-window centroids/labels for trajectory building.
    centroids_by_window: dict[str, np.ndarray] = {}
    l2g_by_window: dict[str, list[int]] = {}
    for w, info in dec_summary.get("windows", {}).items():
        sizes = info["cluster_sizes"]
        n_clusters = len(sizes)
        mask = (panel["window_id"] == w).to_numpy()
        vecs_w = vectors[mask]
        local_assign = assignments.loc[assignments["complaint_id"].isin(
            panel.loc[mask, "complaint_id"]), "local_cluster"]
        cents = np.zeros((n_clusters, vecs_w.shape[1]), dtype=np.float32)
        for c in range(n_clusters):
            sel = vecs_w[(local_assign.to_numpy() == c)]
            if len(sel):
                cents[c] = sel.mean(axis=0)
        centroids_by_window[w] = cents
        l2g_by_window[w] = [info["local_to_global"][str(c)] for c in range(n_clusters)]

    bundle = build_trajectories_from_frames(
        panel, assignments, vectors, centroids_by_window, l2g_by_window
    )

    # Phase 4 ------------------------------------------------------------
    if not bundle.sequences:
        raise RuntimeError("No cluster trajectories built; DEC produced no windows.")
    settled = [
        s for s, m in zip(bundle.sequences, bundle.meta, strict=False)
        if m["length"] >= 3  # trained only on multi-window "settled" trajectories
    ]
    train_seqs = settled or bundle.sequences[: max(1, len(bundle.sequences) // 2)]
    model = train_lstm_ae(train_seqs, cfg=drift_cfg)
    scores = score_trajectories(model, bundle.sequences)
    threshold = choose_threshold(
        score_trajectories(model, train_seqs)["overall"], drift_cfg
    )
    alerts = flag_alerts(scores, threshold)

    ranked = []
    order = sorted(range(len(scores["overall"])), key=lambda j: -scores["overall"][j])
    for i in order:
        ranked.append({
            "rank": len(ranked) + 1,
            "global_cluster": bundle.meta[i]["global_cluster"],
            "first_window": bundle.meta[i]["first_window"],
            "error": scores["overall"][i],
            "per_feature": {
                name: round(float(v), 6)
                for name, v in zip(FEATURE_NAMES, scores["per_feature"][i], strict=False)
            },
            "alert": scores["overall"][i] > threshold,
        })
    save_json({"threshold": threshold, "ranked_clusters": ranked},
              _cfg.OUTPUTS_DIR / "emergent_clusters.json")

    return PipelineResult(
        panel=panel,
        embeddings=vectors,
        assignments=assignments,
        dec_summary=dec_summary,
        trajectories=bundle,
        drift_scores=scores,
        alerts=alerts,
        threshold=threshold,
    )
