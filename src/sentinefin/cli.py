"""Command-line interface for SentinelFin.

Usage:
  sentinefin ingest   [--raw PATH]          Phase 1: panel + EDA
  sentinefin embed                          Phase 2: embeddings + MLP baseline
  sentinefin cluster                        Phase 3: DEC over windows
  sentinefin drift                          Phase 4: LSTM-AE emergence scoring
  sentinefin pipeline [--smoke]             Phases 1-4 end to end
  sentinefin backtest                       Phase 5: pre-registered backtest
  sentinefin report                         Phase 6: static HTML report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinefin", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="Create a small, balanced dataset from complaints.csv")
    p_sample.add_argument("--raw", type=Path, default=None, help="raw source csv/zip path")
    p_sample.add_argument("--output", "-o", type=Path, default=None, help="output small csv path")
    p_sample.add_argument(
        "--size", "-n", type=int, default=20_000, help="target sample size (default 20000)"
    )
    p_sample.add_argument(
        "--min-words", type=int, default=10, help="minimum narrative words (default 10)"
    )
    p_sample.add_argument("--seed", type=int, default=42, help="random seed (default 42)")

    p_ingest = sub.add_parser("ingest", help="Phase 1: build the complaint panel and EDA")
    p_ingest.add_argument("--raw", type=Path, default=None, help="raw csv/zip path override")

    sub.add_parser("embed", help="Phase 2: embeddings + MLP baseline")

    sub.add_parser("cluster", help="Phase 3: DEC clustering over windows")

    sub.add_parser("drift", help="Phase 4: LSTM-AE emergence scoring")

    p_pipe = sub.add_parser("pipeline", help="Phases 1-4 end to end")
    p_pipe.add_argument("--smoke", action="store_true", help="tiny smoke profile")
    p_pipe.add_argument("--skip-eda", action="store_true")

    p_back = sub.add_parser("backtest", help="Phase 5: retrospective backtesting")
    p_back.add_argument("--smoke", action="store_true")

    sub.add_parser("report", help="Phase 6: static HTML report")

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "sample":
        from .ingest import sample_raw_dataset

        out = sample_raw_dataset(
            input_path=args.raw,
            output_path=args.output,
            target_size=args.size,
            min_words=args.min_words,
            seed=args.seed,
        )
        print(f"Sampled dataset created at: {out}")
        return 0

    if args.command == "ingest":
        from .config import DataConfig
        from .ingest import build_panel, load_raw, run_eda

        raw = load_raw(args.raw) if args.raw else load_raw()
        panel = build_panel(raw, config=DataConfig())
        for path in run_eda(panel):
            print(f"wrote {path}")
        print(f"panel: {len(panel)} complaints, {panel['window_id'].nunique()} windows")
        return 0

    if args.command == "embed":
        from .config import EmbeddingConfig, MLPConfig
        from .embed import load_embeddings_state
        from .mlp_baseline import train_mlp_baseline

        vectors, panel = load_embeddings_state(refresh=True, config=EmbeddingConfig())
        metrics = train_mlp_baseline(vectors, panel["product"], config=MLPConfig())
        print(metrics)
        return 0

    if args.command == "cluster":
        from .config import DECConfig
        from .dec import run_dec_over_windows
        from .embed import load_embeddings_state

        vectors, panel = load_embeddings_state()
        _, summary = run_dec_over_windows(panel, vectors, cfg=DECConfig())
        print({w: v.get("silhouette") for w, v in summary["windows"].items()})
        return 0

    if args.command == "drift":
        import json

        from .config import OUTPUTS_DIR, DriftConfig
        from .drift import choose_threshold, score_trajectories, train_lstm_ae
        from .trajectories import build_trajectories_from_frames
        from .utils import load_json

        summary = load_json(OUTPUTS_DIR / "dec_summary.json")
        # Rebuild trajectories from persisted artifacts.
        import numpy as np
        import pandas as pd

        from .config import PROCESSED_DATA_DIR

        panel = pd.read_parquet(PROCESSED_DATA_DIR / "panel.parquet")
        assigns = pd.read_parquet(PROCESSED_DATA_DIR / "dec_assignments.parquet")
        vectors = np.load(PROCESSED_DATA_DIR / "embeddings.npy")
        centroids_by_window, l2g_by_window = _centroids_from_summary(
            panel, vectors, assigns, summary
        )
        bundle = build_trajectories_from_frames(
            panel, assigns, vectors, centroids_by_window, l2g_by_window
        )
        cfg = DriftConfig()
        model = train_lstm_ae(bundle.sequences, cfg=cfg)
        scores = score_trajectories(model, bundle.sequences)
        threshold = choose_threshold(scores["overall"], cfg)
        from .drift import flag_alerts

        alerts = flag_alerts(scores, threshold)
        (OUTPUTS_DIR / "emergent_clusters.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "ranked_clusters": [
                        {
                            "global_cluster": m["global_cluster"],
                            "first_window": m["first_window"],
                            "error": e,
                            "alert": e > threshold,
                        }
                        for m, e in zip(bundle.meta, scores["overall"], strict=False)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{len(alerts)} alerts above threshold {threshold:.4f}")
        return 0

    if args.command == "pipeline":
        from .pipeline import run_pipeline

        result = run_pipeline(smoke=args.smoke, skip_eda=args.skip_eda)
        n_alerts = len(result.alerts)
        print(
            f"pipeline complete: {len(result.panel)} complaints, "
            f"{len(result.dec_summary['windows'])} DEC windows, {n_alerts} alerts"
        )
        return 0

    if args.command == "backtest":
        from .backtest import run_backtest

        report = run_backtest(smoke=args.smoke)
        for case in report["cases"]:
            print(case)
        return 0

    if args.command == "report":
        from .reporting import build_report

        path = build_report()
        print(f"report: {path}")
        return 0

    return 1


def _centroids_from_summary(panel, vectors, assigns, summary):
    """Recompute per-window mean centroids from persisted assignments."""
    import numpy as np

    centroids_by_window: dict[str, object] = {}
    l2g_by_window: dict[str, list[int]] = {}
    merged = assigns.merge(panel[["complaint_id", "window_id"]], on="complaint_id")
    for w, info in summary.get("windows", {}).items():
        sub = merged[merged["window_id"] == w]
        n_clusters = len(info["cluster_sizes"])
        mask = (panel["window_id"] == w).to_numpy()
        vecs_w = vectors[mask]
        id_pos = {cid: i for i, cid in enumerate(panel.loc[mask, "complaint_id"])}
        cents = np.zeros((n_clusters, vecs_w.shape[1]), dtype=np.float32)
        for c in range(n_clusters):
            ids = sub.loc[sub["local_cluster"] == c, "complaint_id"]
            pos = [id_pos[cid] for cid in ids if cid in id_pos]
            if pos:
                cents[c] = vecs_w[pos].mean(axis=0)
        centroids_by_window[w] = cents
        l2g_by_window[w] = [info["local_to_global"][str(c)] for c in range(n_clusters)]
    return centroids_by_window, l2g_by_window


if __name__ == "__main__":
    sys.exit(main())
