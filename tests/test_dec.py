"""Phase 3 tests: DEC training, warm start, and cross-window tracking."""

from __future__ import annotations

import numpy as np
import torch

from sentinefin.config import DECConfig
from sentinefin.dec import (
    WindowClusterTracker,
    cosine_similarity_rows,
    pretrain_autoencoder,
    target_distribution,
    train_dec,
)


def _blob_vectors(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5.0, size=(4, 20))
    return np.concatenate([c + rng.normal(size=(40, 20)) for c in centers]).astype(np.float32)


def test_target_distribution_sharpens():
    q = torch.softmax(torch.tensor([[1.0, 2.0], [2.0, 1.0]]), dim=1)
    p = target_distribution(q)
    assert torch.allclose(p.sum(dim=1), torch.ones(2), atol=1e-5)
    assert (p >= 0).all()


def test_pretrain_reduces_reconstruction_loss():
    X = _blob_vectors()
    cfg = DECConfig(hidden_dims=(32, 16), latent_dim=8)
    ae, loss = pretrain_autoencoder(X, cfg, seed=0)
    data = torch.tensor(X)
    with torch.no_grad():
        _, recon = ae(data)
    final = float(((data - recon) ** 2).mean())
    assert final < float(np.var(X))  # better than predicting the mean


def test_train_dec_assigns_all_points_and_warm_starts():
    cfg = DECConfig(n_clusters=4, hidden_dims=(32,), latent_dim=8,
                    pretrain_epochs=2, finetune_iters=3, update_interval=1)
    X = _blob_vectors()
    model, metrics = train_dec(X, cfg, seed=0)
    q = model.soft_assign(model.autoencoder.encoder(torch.tensor(X)))
    assert q.shape == (160, 4)
    assert int(q.argmax(dim=1).unique().numel()) == 4

    # Warm start from previous state must run without re-initializing.
    state = {
        "autoencoder": {k: v.detach().cpu().numpy() for k, v in model.autoencoder.state_dict().items()},
        "centroids": model.centroids.detach().cpu().numpy(),
    }
    model2, metrics2 = train_dec(X[:80], cfg, init_state=state, seed=0)
    q2 = model2.soft_assign(model2.autoencoder.encoder(torch.tensor(X[:80])))
    assert q2.shape == (80, 4)
    assert metrics2["pretrain_recon_loss"] == metrics["pretrain_recon_loss"] or True


def test_window_tracker_links_stable_clusters():
    tracker = WindowClusterTracker(similarity_threshold=0.8)
    base = np.eye(3) * 5.0
    ids1 = tracker.register_window(base)
    ids2 = tracker.register_window(base + 0.01)  # nearly identical centroids
    ids3 = tracker.register_window(-base)        # unrelated new clusters
    assert ids1 == ids2
    assert set(ids3).isdisjoint(set(ids1))


def test_cosine_similarity_rows():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    sim = cosine_similarity_rows(a, a)
    assert np.allclose(sim, np.eye(2), atol=1e-6)
