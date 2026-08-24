"""Phase 2 tests: offline fallback encoder and MLP baseline."""

from __future__ import annotations

import numpy as np

from conftest_data import synthetic_complaints
from sentinefin.config import EmbeddingConfig, MLPConfig
from sentinefin.embed import HashedBoWEncoder, embed_panel, load_encoder
from sentinefin.mlp_baseline import train_mlp_baseline


def test_hashed_encoder_is_deterministic_and_normalized():
    enc = HashedBoWEncoder(dim=64)
    texts = ["credit report dispute error", "mortgage escrow payment", "credit report dispute error"]
    out1 = enc.encode(texts)
    out2 = enc.encode(texts)
    assert out1.shape == (3, 64)
    assert np.allclose(out1[0], out1[2])
    assert np.allclose(out1[0], out2[0])
    assert not np.allclose(out1[0], out1[1])
    norms = np.linalg.norm(out1, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_load_encoder_falls_back_offline(monkeypatch):
    cfg = EmbeddingConfig()
    cfg.hash_dim = 32

    def boom(name):
        raise RuntimeError("no network in CI")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", boom)
    enc, backend = load_encoder(cfg)
    assert backend == "hashed-bow"
    assert enc.encode(["hello world"]).shape == (1, 32)


def test_embed_panel_persists_artifacts(isolated_dirs):
    from sentinefin import config as cfgmod

    df = synthetic_complaints(n_per_topic=12)
    # Build panel directly into the isolated dir.
    from sentinefin.config import DataConfig
    from sentinefin.ingest import build_panel

    panel = build_panel(df, config=DataConfig(),
                        processed_dir=cfgmod.PROCESSED_DATA_DIR)
    vectors = embed_panel(panel, config=EmbeddingConfig(),
                          processed_dir=cfgmod.PROCESSED_DATA_DIR)
    assert vectors.shape[0] == len(panel)
    assert (cfgmod.PROCESSED_DATA_DIR / "embeddings.npy").exists()
    assert (cfgmod.PROCESSED_DATA_DIR / "embeddings_meta.json").exists()


def test_mlp_baseline_beats_chance(isolated_dirs):
    from sentinefin import config as cfgmod

    rng = np.random.default_rng(0)
    n_classes, per_class = 3, 60
    centers = rng.normal(scale=4.0, size=(n_classes, 16))
    X = np.concatenate([c + rng.normal(size=(per_class, 16)) for c in centers]).astype(np.float32)
    y = np.repeat([f"class_{i}" for i in range(n_classes)], per_class)

    metrics = train_mlp_baseline(
        X,
        y,
        config=MLPConfig(hidden_dims=(32,), epochs=8, batch_size=32),
        seed=0,
        outputs_dir=cfgmod.OUTPUTS_DIR,
    )
    assert metrics["accuracy"] > 1.0 / n_classes + 0.3
