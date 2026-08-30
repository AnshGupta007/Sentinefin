"""Phase 2: sentence embeddings with a deterministic offline fallback.

Primary encoder: a pretrained sentence-transformers Transformer
(all-MiniLM-L6-v2). When the model cannot be downloaded (air-gapped CI), we
degrade gracefully to a deterministic hashed bag-of-words embedder so every
downstream neural stage can still be exercised end-to-end.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as _cfg
from .config import EmbeddingConfig
from .ingest import NARRATIVE_COL
from .utils import ensure_dir, stable_hash

logger = logging.getLogger(__name__)


def embeddings_paths() -> tuple[Path, Path, Path]:
    """Resolve artifact paths at call time so test isolation patches apply."""
    base = _cfg.PROCESSED_DATA_DIR
    return base / "embeddings.npy", base / "embedding_ids.parquet", base / "embeddings_meta.json"


class HashedBoWEncoder:
    """Deterministic offline stand-in for the Transformer encoder.

    Not used for reported research results; exists so the full pipeline is
    runnable and testable without network access.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in str(text).lower().split():
                idx = stable_hash(tok) % self.dim
                sign = 1.0 if stable_hash("s:" + tok) % 2 == 0 else -1.0
                vecs[i, idx] += sign
            norm = float(np.linalg.norm(vecs[i]))
            if norm > 0:
                vecs[i] /= norm
        return vecs


def load_encoder(config: EmbeddingConfig):
    """Try the pretrained Transformer first; fall back to hashed BoW."""
    import os
    if os.environ.get("SENTINEFIN_FAST_EMBED", "0") == "1":
        logger.info("SENTINEFIN_FAST_EMBED=1 set; using HashedBoWEncoder(%d).", config.hash_dim)
        return HashedBoWEncoder(config.hash_dim), "hashed-bow"
    try:
        from sentence_transformers import SentenceTransformer

        enc = SentenceTransformer(config.model_name)
        logger.info("Loaded pretrained encoder %s", config.model_name)
        return enc, config.model_name
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Pretrained encoder unavailable (%s); using hashed BoW.", exc)
        return HashedBoWEncoder(config.hash_dim), "hashed-bow"


def embed_panel(panel: pd.DataFrame, config: EmbeddingConfig | None = None,
                processed_dir: Path | None = None) -> np.ndarray:
    """Embed all narratives; persist embeddings + id index to disk."""
    cfg = config or EmbeddingConfig()
    processed_dir = processed_dir or _cfg.PROCESSED_DATA_DIR

    from .utils import set_seed

    set_seed(cfg.seed)
    encoder, backend = load_encoder(cfg)
    texts = panel[NARRATIVE_COL].astype(str).tolist()
    logger.info("Embedding %d narratives with %s ...", len(texts), backend)
    vectors = encoder.encode(texts, batch_size=cfg.batch_size, show_progress_bar=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    ensure_dir(processed_dir)
    np.save(processed_dir / "embeddings.npy", vectors)
    panel[["complaint_id"]].to_parquet(processed_dir / "embedding_ids.parquet", index=False)
    save_meta(
        processed_dir,
        {
            "backend": backend,
            "model_name": cfg.model_name,
            "dim": int(vectors.shape[1]),
            "n": int(vectors.shape[0]),
        },
    )
    return vectors


def save_meta(processed_dir: Path, meta: dict) -> Path:
    import json

    path = processed_dir / "embeddings_meta.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def load_embeddings(processed_dir: Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Load saved embeddings aligned to their complaint ids."""
    processed_dir = processed_dir or _cfg.PROCESSED_DATA_DIR
    vectors = np.load(processed_dir / "embeddings.npy")
    ids = pd.read_parquet(processed_dir / "embedding_ids.parquet")
    return vectors, ids


def load_embeddings_state(refresh: bool = False, config: EmbeddingConfig | None = None,
                          processed_dir: Path | None = None):
    """Return (vectors, panel), embedding fresh and rebuilding the panel if needed.

    Used by stage-wise CLI commands so each phase can run independently while
    reusing persisted artifacts from earlier phases.
    """
    from .ingest import build_panel, load_raw

    target_dir = processed_dir or _cfg.PROCESSED_DATA_DIR
    if refresh or not target_dir.joinpath("embeddings.npy").exists():
        panel_path = target_dir / "panel.parquet"
        if panel_path.exists():
            panel = pd.read_parquet(panel_path)
        else:
            raw = load_raw()
            panel = build_panel(raw)
        vectors = embed_panel(panel, config=config, processed_dir=target_dir)
        return vectors, panel
    vectors = np.load(target_dir / "embeddings.npy")
    panel = pd.read_parquet(target_dir / "panel.parquet")
    return vectors, panel
