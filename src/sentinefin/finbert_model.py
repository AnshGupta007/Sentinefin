"""FinBERT domain-specific language model for financial complaint classification.

Uses 'ProsusAI/finbert', a BERT model pre-trained on corporate financial filings
and financial sentiment data, to produce contextual financial representations and
classify consumer grievances into canonical regulatory categories.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from . import config as _cfg
from .model_evaluator import compute_metrics
from .utils import set_seed

logger = logging.getLogger(__name__)

DEFAULT_FINBERT_NAME = "ProsusAI/finbert"


class FinBERTClassifierHead(nn.Module):
    """Deep non-linear classification head over 768-d FinBERT embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, n_classes: int = 11, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_finbert_embeddings(
    texts: list[str],
    model_name: str = DEFAULT_FINBERT_NAME,
    batch_size: int = 32,
    max_length: int = 128,
    cache_path: Path | str | None = None,
) -> np.ndarray:
    """Extract 768-d [CLS] pooled representations from FinBERT with optional caching."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Loading cached FinBERT embeddings from %s", cache_path)
            return np.load(cache_path)

    device = torch.device("cpu")
    logger.info("Loading %s tokenizer and transformer backbone...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    transformer = AutoModel.from_pretrained(model_name).to(device)
    transformer.eval()

    all_embeddings = []
    n = len(texts)
    logger.info("Extracting FinBERT embeddings for %d complaint texts (batch_size=%d)...", n, batch_size)

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            batch_texts = texts[start_idx : start_idx + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            outputs = transformer(**encoded)
            # Use pooled output or mean pooled representation
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                cls_embeds = outputs.pooler_output
            else:
                cls_embeds = outputs.last_hidden_state[:, 0, :]

            all_embeddings.append(cls_embeds.cpu().numpy())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info("Extracted FinBERT embeddings matrix shape: %s", embeddings.shape)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info("Cached FinBERT embeddings to %s", cache_path)

    return embeddings


def train_finbert_classifier(
    train_texts: list[str],
    y_train: np.ndarray,
    test_texts: list[str],
    y_test: np.ndarray,
    classes: list[str],
    model_name: str = DEFAULT_FINBERT_NAME,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.002,
    seed: int = 17,
    cache_dir: Path | str | None = None,
) -> tuple[FinBERTClassifierHead, dict[str, Any]]:
    """Extract FinBERT representations, train the deep classification head, and evaluate."""
    set_seed(seed)
    start_time = time.time()
    device = torch.device("cpu")

    cache_dir = Path(cache_dir or (_cfg.PROCESSED_DATA_DIR / "finbert_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    tr_cache = cache_dir / "train_finbert_embeddings.npy"
    te_cache = cache_dir / "test_finbert_embeddings.npy"

    # 1. Extract or load FinBERT embeddings
    X_train_emb = extract_finbert_embeddings(
        train_texts, model_name=model_name, batch_size=32, cache_path=tr_cache
    )
    X_test_emb = extract_finbert_embeddings(
        test_texts, model_name=model_name, batch_size=32, cache_path=te_cache
    )

    # 2. Setup training dataset
    X_tr_t = torch.tensor(X_train_emb, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_emb, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 3. Model & Optimizer
    n_classes = len(classes)
    model = FinBERTClassifierHead(
        input_dim=X_train_emb.shape[1], hidden_dim=256, n_classes=n_classes, dropout=0.2
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    # 4. Training loop
    logger.info("Training FinBERT classification head for %d epochs...", epochs)
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            opt.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
        scheduler.step()

    train_time = time.time() - start_time

    # 5. Evaluation
    inf_start = time.time()
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_t)
        preds = test_logits.argmax(dim=1).cpu().numpy()
    inf_time = time.time() - inf_start

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=preds,
        classes=classes,
        train_time_sec=train_time,
        inference_time_sec=inf_time,
    )
    metrics["model_name"] = "FinBERT (ProsusAI)"
    metrics["backbone"] = model_name
    metrics["embedding_dim"] = int(X_train_emb.shape[1])
    metrics["epochs"] = epochs

    logger.info(
        "FinBERT: Accuracy=%.3f (%.1f%%), Macro-F1=%.3f",
        metrics["accuracy"],
        metrics["accuracy"] * 100,
        metrics["macro_f1"],
    )

    return model, metrics
