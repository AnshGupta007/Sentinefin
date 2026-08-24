"""Phase 4: LSTM Autoencoder for cluster-trajectory drift/emergence scoring.

Each cluster's history is a time series of per-window feature vectors
(centroid movement, complaint volume, size share, mean distance to centroid).
An LSTM autoencoder is trained on trajectories from "settled" clusters so it
learns what normal trajectory patterns look like. At inference the
reconstruction error becomes the emergence signal: a trajectory that
reconstructs poorly is behaving in a way the model has not learned to expect,
i.e. a candidate emergent pattern. Scores decompose into interpretable
sub-components (volume vs. shape dimensions), and a configurable threshold
flags candidate alerts.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from .config import DriftConfig
from .utils import set_seed

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["centroid_shift", "volume", "volume_share", "mean_dist"]


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> latent vector -> decoder LSTM (reconstructs the sequence)."""

    def __init__(self, n_features: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.encoder(x)
        z = self.to_latent(h_n[-1])
        h0 = self.from_latent(z).unsqueeze(0)
        c0 = torch.zeros_like(h0)
        dec_in = h0.transpose(0, 1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_in, (h0, c0))
        recon = self.output(dec_out)
        return recon, z


def per_feature_error(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Mean squared error per feature per timestep; keeps sub-components interpretable."""
    return ((x - recon) ** 2).mean(dim=1)  # (batch, n_features)


def train_lstm_ae(
    sequences: list[np.ndarray],
    cfg: DriftConfig | None = None,
    seed: int = 31,
) -> LSTMAutoencoder:
    """Train on 'settled' trajectories; pads short ones via repetition."""
    cfg = cfg or DriftConfig()
    set_seed(seed)
    n_features = sequences[0].shape[1]
    model = LSTMAutoencoder(n_features, cfg.hidden_dim, cfg.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss(reduction="none")
    model.train()
    for epoch in range(cfg.epochs):
        total = 0.0
        for seq in sequences:
            x = torch.tensor(pad_or_trim(seq, min_len=3), dtype=torch.float32).unsqueeze(0)
            opt.zero_grad()
            recon, _ = model(x)
            loss = mse(x, recon).mean()
            loss.backward()
            opt.step()
            total += float(loss.item())
        if (epoch + 1) % max(1, cfg.epochs // 5) == 0:
            logger.info("LSTM-AE epoch %d/%d loss %.5f", epoch + 1, cfg.epochs, total / len(sequences))
    return model


def pad_or_trim(seq: np.ndarray, min_len: int = 3, max_len: int = 24) -> np.ndarray:
    """Pad too-short trajectories by repeating their first step; cap length."""
    if len(seq) < min_len:
        reps = int(np.ceil(min_len / len(seq)))
        seq = np.tile(seq[:1], (reps, 1))[:max(min_len, len(seq))]
    if len(seq) > max_len:
        seq = seq[-max_len:]
    return seq


@torch.no_grad()
def score_trajectories(
    model: LSTMAutoencoder,
    sequences: list[np.ndarray],
) -> dict[str, list[float]]:
    """Reconstruction-error scores plus interpretable per-feature components."""
    model.eval()
    overall: list[float] = []
    per_feature: list[list[float]] = []
    for seq in sequences:
        x = torch.tensor(pad_or_trim(seq), dtype=torch.float32).unsqueeze(0)
        recon, _ = model(x)
        errs = ((x - recon) ** 2).mean(dim=(0, 1)).tolist()  # per-feature mean
        overall.append(float(np.mean(errs)))
        per_feature.append(errs)
    return {"overall": overall, "per_feature": per_feature}


def choose_threshold(train_errors: list[float], cfg: DriftConfig) -> float:
    """Alert cut at the configured quantile of normal-trajectory training error."""
    q = float(np.quantile(np.asarray(train_errors), cfg.quantile))
    return float(q * (1.0 + cfg.margin))


def flag_alerts(scores: dict, threshold: float) -> list[dict]:
    alerts = []
    for i, err in enumerate(scores["overall"]):
        if err > threshold:
            alerts.append(
                {
                    "trajectory_index": i,
                    "error": err,
                    "per_feature": {
                        name: err_f for name, err_f in zip(FEATURE_NAMES, scores["per_feature"][i], strict=False)
                    },
                }
            )
    return sorted(alerts, key=lambda a: -a["error"])
