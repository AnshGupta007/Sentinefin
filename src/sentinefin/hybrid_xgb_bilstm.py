"""Hybrid XGBoost + BiLSTM model for CFPB complaint classification.

Combines a deep Bidirectional LSTM (BiLSTM) feature extractor with an Extreme
Gradient Boosting (XGBoost) ensemble classifier. BiLSTM models contextual sequential
dependencies across representations, and XGBoost provides high-capacity, non-linear
decision boundary partitioning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from .model_evaluator import compute_metrics
from .utils import set_seed

logger = logging.getLogger(__name__)


class BiLSTMFeatureExtractor(nn.Module):
    """Bidirectional LSTM neural network for extracting high-order deep features."""

    def __init__(
        self,
        input_dim: int = 32,
        seq_len: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 11,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2 directions * hidden_dim = 2 * hidden_dim
        lstm_out_dim = 2 * hidden_dim
        # Global pooling (avg + max) = 2 * lstm_out_dim = 4 * hidden_dim
        self.feature_dim = 2 * lstm_out_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run through BiLSTM and pool over sequence length."""
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, 2 * hidden_dim)

        avg_pool = torch.mean(lstm_out, dim=1)  # (batch_size, 2 * hidden_dim)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch_size, 2 * hidden_dim)
        features = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, 4 * hidden_dim)
        return features

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits, features


def prepare_sequential_tensor(vectors: np.ndarray, seq_len: int = 12) -> torch.Tensor:
    """Reshape flat 384-d vectors into a sequential tensor of shape (N, seq_len, 384 // seq_len)."""
    n, total_dim = vectors.shape
    assert total_dim % seq_len == 0, f"Cannot divide {total_dim} evenly into {seq_len} sequence steps"
    feat_dim = total_dim // seq_len
    reshaped = vectors.reshape(n, seq_len, feat_dim)
    return torch.tensor(reshaped, dtype=torch.float32)


def train_hybrid_xgb_bilstm(
    X_train_vecs: np.ndarray,
    y_train: np.ndarray,
    X_test_vecs: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    epochs_bilstm: int = 25,
    batch_size: int = 64,
    bilstm_lr: float = 0.002,
    xgb_estimators: int = 120,
    xgb_depth: int = 5,
    seed: int = 17,
) -> tuple[xgb.XGBClassifier, BiLSTMFeatureExtractor, dict[str, Any]]:
    """Train the Hybrid XGBoost + BiLSTM pipeline and compute evaluation metrics."""
    set_seed(seed)
    start_time = time.time()
    device = torch.device("cpu")

    seq_len = 12
    feat_dim = X_train_vecs.shape[1] // seq_len
    n_classes = len(classes)

    # 1. Prepare sequential inputs for BiLSTM
    X_tr_seq = prepare_sequential_tensor(X_train_vecs, seq_len=seq_len)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long)

    X_te_seq = prepare_sequential_tensor(X_test_vecs, seq_len=seq_len)
    y_te_tensor = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_tr_seq, y_tr_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # 2. Instantiate and train BiLSTM feature extractor
    bilstm = BiLSTMFeatureExtractor(
        input_dim=feat_dim,
        seq_len=seq_len,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        n_classes=n_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(bilstm.parameters(), lr=bilstm_lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("Training BiLSTM feature extractor for %d epochs...", epochs_bilstm)
    bilstm.train()
    for epoch in range(epochs_bilstm):
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, _ = bilstm(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

    # 3. Extract latent deep features from BiLSTM for train and test
    bilstm.eval()
    with torch.no_grad():
        _, tr_deep_features = bilstm(X_tr_seq)
        _, te_deep_features = bilstm(X_te_seq)

    tr_deep_np = tr_deep_features.cpu().numpy()
    te_deep_np = te_deep_features.cpu().numpy()

    # 4. Fuse representations: original 384-d semantic embeddings + 256-d BiLSTM deep features
    X_tr_hybrid = np.hstack([X_train_vecs, tr_deep_np])
    X_te_hybrid = np.hstack([X_test_vecs, te_deep_np])

    logger.info(
        "Fused hybrid feature dimension: %d (384 embeddings + %d BiLSTM features)",
        X_tr_hybrid.shape[1],
        tr_deep_np.shape[1],
    )

    # 5. Train XGBoost ensemble classifier on fused representations
    logger.info("Training XGBoost ensemble classifier with %d estimators...", xgb_estimators)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=xgb_estimators,
        max_depth=xgb_depth,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=seed,
        n_jobs=4,
        eval_metric="mlogloss",
    )
    xgb_clf.fit(X_tr_hybrid, y_train)

    train_time = time.time() - start_time

    # 6. Evaluation and timing
    inf_start = time.time()
    preds = xgb_clf.predict(X_te_hybrid)
    inf_time = time.time() - inf_start

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=preds,
        classes=classes,
        train_time_sec=train_time,
        inference_time_sec=inf_time,
    )
    metrics["model_name"] = "Hybrid XGBoost + BiLSTM"
    metrics["fused_feature_dim"] = int(X_tr_hybrid.shape[1])
    metrics["xgb_estimators"] = xgb_estimators
    metrics["bilstm_epochs"] = epochs_bilstm

    logger.info(
        "Hybrid XGBoost + BiLSTM: Accuracy=%.3f (%.1f%%), Macro-F1=%.3f",
        metrics["accuracy"],
        metrics["accuracy"] * 100,
        metrics["macro_f1"],
    )

    return xgb_clf, bilstm, metrics
