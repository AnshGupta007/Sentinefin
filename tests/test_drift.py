"""Phase 4 tests: LSTM-AE scoring, thresholds, alerts on synthetic trajectories."""

from __future__ import annotations

import numpy as np

from sentinefin.config import DriftConfig
from sentinefin.drift import (
    LSTMAutoencoder,
    choose_threshold,
    flag_alerts,
    pad_or_trim,
    score_trajectories,
    train_lstm_ae,
)


def _normal_traj(rng, n=6):
    # Smooth, slowly-varying trajectory the AE should reconstruct easily.
    t = np.linspace(0, 1, n)
    base = np.stack([0.5 + 0.05 * np.sin(2 * np.pi * t + p) for p in (0.0, 1.0, 2.0)], axis=1)
    return (base + rng.normal(scale=0.01, size=base.shape)).astype(np.float32)


def test_pad_or_trim():
    short = np.ones((1, 4), dtype=np.float32)
    padded = pad_or_trim(short, min_len=3, max_len=24)
    assert padded.shape[0] == 3
    long = np.ones((30, 4), dtype=np.float32)
    assert pad_or_trim(long, min_len=3, max_len=24).shape[0] == 24


def test_lstm_ae_flags_anomalous_trajectory():
    cfg = DriftConfig(hidden_dim=16, latent_dim=8, epochs=80, lr=5e-3)
    rng = np.random.default_rng(0)
    normal = [_normal_traj(rng) for _ in range(12)]
    model = train_lstm_ae(normal, cfg=cfg, seed=0)

    anomalous = np.stack(
        [np.linspace(0.9, 9.0, 6)] * 3, axis=1
    ).astype(np.float32)

    scores = score_trajectories(model, normal + [anomalous])
    normal_errs = scores["overall"][:-1]
    anomaly_err = scores["overall"][-1]
    assert anomaly_err > max(normal_errs) * 3.0


def test_threshold_and_alerts():
    cfg = DriftConfig(quantile=0.9, margin=0.1)
    train_errors = [1.0, 1.1, 0.9, 1.05, 1.0, 0.95]
    threshold = choose_threshold(train_errors, cfg)
    scores = {
        "overall": [1.0, threshold * 5],
        "per_feature": [[0.25] * 4, [5.0, 5.0, 5.0, 5.0]],
    }
    alerts = flag_alerts(scores, threshold)
    assert len(alerts) == 1
    assert alerts[0]["trajectory_index"] == 1
    assert "volume" in alerts[0]["per_feature"]


def test_model_shapes():
    import torch

    m = LSTMAutoencoder(n_features=4, hidden_dim=8, latent_dim=4)
    x = torch.randn(2, 6, 4)
    recon, z = m(x)
    assert recon.shape == x.shape
    assert z.shape == (2, 4)
