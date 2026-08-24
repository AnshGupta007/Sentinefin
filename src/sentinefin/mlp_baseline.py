"""Phase 2: neural MLP baseline classifier over complaint embeddings.

Trains a real feedforward neural network (not logistic regression) to predict
the complaint product category from embeddings. This sanity-checks that the
representations carry usable signal before the DEC and LSTM-AE stages.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from . import config as _cfg
from .config import MLPConfig
from .utils import save_json, set_seed

logger = logging.getLogger(__name__)


class MLPTagClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dims: tuple[int, ...],
                 dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_baseline(
    vectors: np.ndarray,
    labels_raw: pd.Series,
    config: MLPConfig | None = None,
    seed: int = 17,
    outputs_dir=None,
) -> dict:
    cfg = config or MLPConfig()
    set_seed(seed)

    le = LabelEncoder()
    y = le.fit_transform(labels_raw.astype(str))
    mask = pd.Series(labels_raw).notna().to_numpy()
    X, y_arr = vectors[mask], y[mask]

    (X_tr, X_te, y_tr, y_te) = train_test_split(
        X, y_arr, test_size=0.2, random_state=seed, stratify=y_arr
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPTagClassifier(
        input_dim=X.shape[1],
        n_classes=len(le.classes_),
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

    n = len(X_tr_t)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            opt.zero_grad()
            logits = model(X_tr_t[idx])
            loss = loss_fn(logits, y_tr_t[idx])
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx)
        logger.info("MLP epoch %d/%d loss %.4f", epoch + 1, cfg.epochs, total_loss / max(n, 1))

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        preds = logits.argmax(dim=1).cpu().numpy()
    acc = float((preds == y_te).mean())
    macro_f1 = float(f1_score(y_te, preds, average="macro", zero_division=0))
    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_classes": int(len(le.classes_)),
        "hidden_dims": list(cfg.hidden_dims),
        "epochs": cfg.epochs,
    }
    logger.info("Baseline MLP accuracy=%.3f macro-F1=%.3f", acc, macro_f1)
    outputs_dir = outputs_dir or _cfg.OUTPUTS_DIR
    save_json(metrics, outputs_dir / "mlp_baseline_metrics.json")
    return metrics
