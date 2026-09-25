"""Unified evaluation utilities and metrics for model comparison on complaints_small.csv."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config as _cfg
from .mlp_baseline import canonicalize_product_labels
from .utils import set_seed

logger = logging.getLogger(__name__)


def load_small_dataset_splits(
    csv_path: Path | str | None = None,
    test_size: float = 0.2,
    seed: int = 17,
) -> dict[str, Any]:
    """Load complaints_small.csv and return stratified train/test partitions."""
    set_seed(seed)
    if csv_path is None:
        csv_path = _cfg.RAW_DATA_DIR / "complaints_small.csv"
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Small dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded %d complaints from %s", len(df), csv_path)

    # Clean product labels
    product_col = next((c for c in df.columns if c.strip().lower() == "product"), "Product")
    df["canonical_product"] = canonicalize_product_labels(df[product_col])

    # Construct contextual text: Issue + Narrative
    issue_col = next((c for c in df.columns if c.strip().lower() == "issue"), "Issue")
    narr_col = next(
        (c for c in df.columns if "narrative" in c.lower()),
        "Consumer complaint narrative",
    )

    df["issue_clean"] = df[issue_col].fillna("General Inquiry").astype(str).str.strip()
    df["narrative_clean"] = df[narr_col].fillna("").astype(str).str.strip()
    df["enriched_text"] = "Issue: " + df["issue_clean"] + ". Narrative: " + df["narrative_clean"]

    # Filter out empty narratives if any
    valid_mask = df["narrative_clean"].str.len() > 0
    df = df[valid_mask].reset_index(drop=True)

    le = LabelEncoder()
    df["label_idx"] = le.fit_transform(df["canonical_product"])
    classes = list(le.classes_)

    train_idx, test_idx = train_test_split(
        df.index.to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=df["label_idx"].to_numpy(),
    )

    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    logger.info(
        "Split dataset into %d train (80%%) and %d test (20%%) across %d classes",
        len(train_df),
        len(test_df),
        len(classes),
    )

    # Load and align sentence embeddings if available
    emb_path = _cfg.PROCESSED_DATA_DIR / "embeddings.npy"
    emb_ids_path = _cfg.PROCESSED_DATA_DIR / "embedding_ids.parquet"
    X_train_vecs = None
    X_test_vecs = None
    aligned_embeddings = None

    if emb_path.exists() and emb_ids_path.exists():
        raw_emb = np.load(emb_path)
        emb_ids = pd.read_parquet(emb_ids_path)
        id_col = next((c for c in df.columns if "complaint" in c.lower() and "id" in c.lower()), None)
        if id_col is not None and "complaint_id" in emb_ids.columns:
            id_to_pos = {cid: i for i, cid in enumerate(emb_ids["complaint_id"])}
            mapped_indices = df[id_col].map(id_to_pos).to_numpy()
            aligned_embeddings = raw_emb[mapped_indices]
            X_train_vecs = aligned_embeddings[train_idx]
            X_test_vecs = aligned_embeddings[test_idx]

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "classes": classes,
        "label_encoder": le,
        "embeddings": aligned_embeddings,
        "X_train_vecs": X_train_vecs,
        "X_test_vecs": X_test_vecs,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    train_time_sec: float = 0.0,
    inference_time_sec: float = 0.0,
) -> dict[str, Any]:
    """Compute comprehensive accuracy, macro-F1, weighted-F1, and per-class stats."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    per_class = {}
    for i, cls_name in enumerate(classes):
        if i < len(precision_arr):
            per_class[cls_name] = {
                "precision": float(precision_arr[i]),
                "recall": float(recall_arr[i]),
                "f1": float(f1_arr[i]),
                "support": int(support_arr[i]),
            }

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_time_sec": round(train_time_sec, 2),
        "inference_time_sec": round(inference_time_sec, 4),
        "n_test": int(len(y_true)),
        "n_classes": len(classes),
        "per_class": per_class,
    }
