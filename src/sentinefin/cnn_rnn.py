"""Proposed CNN-RNN (Convolutional Neural Network + Bidirectional RNN) model.

Integrates multi-scale 1D Convolutional feature extractors (capturing localized
n-gram patterns) with a deep Bidirectional LSTM (capturing global sequential
dependencies), followed by a dense classification head.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model_evaluator import compute_metrics
from .utils import set_seed

logger = logging.getLogger(__name__)


class ProposedCNNRNN(nn.Module):
    """Deep Hybrid CNN-RNN combining multi-scale 1D Conv filters and BiLSTM."""

    def __init__(
        self,
        input_length: int = 384,
        conv_channels: int = 64,
        rnn_hidden_dim: int = 128,
        rnn_layers: int = 2,
        n_classes: int = 11,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = input_length

        # Multi-scale 1D Convolutions with kernel sizes 3, 5, 7
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(16),
        )

        # Fused convolutional channel dimension
        fused_channels = conv_channels * 3

        # Bidirectional Recurrent Layer (BiLSTM)
        self.rnn = nn.LSTM(
            input_size=fused_channels,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Feature dimension after BiLSTM pooling: 2 directions * rnn_hidden_dim * 2 (avg + max)
        self.fc_in_dim = 2 * (2 * rnn_hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(self.fc_in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_length) -> reshape to (batch_size, 1, input_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 1. Multi-scale Convolutional feature extraction
        c3 = self.conv3(x)  # (batch, conv_channels, length / 2)
        c5 = self.conv5(x)  # (batch, conv_channels, length / 2)
        c7 = self.conv7(x)  # (batch, conv_channels, length / 2)

        # Concatenate along channel dimension -> (batch, fused_channels, length / 2)
        fused_conv = torch.cat([c3, c5, c7], dim=1)

        # 2. Transpose for RNN: (batch, seq_len = length / 2, in_features = fused_channels)
        rnn_in = fused_conv.transpose(1, 2)
        rnn_out, _ = self.rnn(rnn_in)  # (batch, seq_len, 2 * rnn_hidden_dim)

        # 3. Global Pooling: Mean-pooling + Max-pooling across time steps
        avg_p = torch.mean(rnn_out, dim=1)
        max_p, _ = torch.max(rnn_out, dim=1)
        pooled = torch.cat([avg_p, max_p], dim=1)  # (batch, fc_in_dim)

        # 4. Dense Classification
        logits = self.classifier(pooled)
        return logits


def train_proposed_cnn_rnn(
    X_train_vecs: np.ndarray,
    y_train: np.ndarray,
    X_test_vecs: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 17,
) -> tuple[ProposedCNNRNN, dict[str, Any]]:
    """Train the Proposed CNN-RNN architecture and compute evaluation metrics."""
    set_seed(seed)
    start_time = time.time()
    device = torch.device("cpu")

    n_classes = len(classes)
    input_length = X_train_vecs.shape[1]

    X_tr_t = torch.tensor(X_train_vecs, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_vecs, dtype=torch.float32)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = ProposedCNNRNN(
        input_length=input_length,
        conv_channels=64,
        rnn_hidden_dim=96,
        rnn_layers=1,
        n_classes=n_classes,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    logger.info("Training Proposed CNN-RNN for %d epochs...", epochs)
    for _epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
        scheduler.step()

    train_time = time.time() - start_time

    # Evaluation
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
    metrics["model_name"] = "Proposed CNN-RNN"
    metrics["conv_channels"] = 64
    metrics["rnn_hidden_dim"] = 128
    metrics["epochs"] = epochs

    logger.info(
        "Proposed CNN-RNN: Accuracy=%.3f (%.1f%%), Macro-F1=%.3f",
        metrics["accuracy"],
        metrics["accuracy"] * 100,
        metrics["macro_f1"],
    )

    return model, metrics
