"""Central configuration for SentinelFin.

All knobs live here so the pipeline is reproducible and CI-friendly. The
smoke-test profile (SENTINEFIN_SMOKE=1) shrinks every model and epoch count so
the whole pipeline runs in a couple of minutes, which is what CI uses.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# config.py lives at <root>/src/sentinefin/config.py -> two levels up is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

CFPB_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


SMOKE: bool = _env_flag("SENTINEFIN_SMOKE")


def _profile(smoke_val, full_val):
    """Evaluate the smoke/full value at CONFIG-INSTANTIATION time.

    Import-time evaluation would freeze the profile for the whole process
    (breaking tests that set SENTINEFIN_SMOKE after other modules imported
    config), so every smoke-dependent default resolves lazily.
    """
    return field(default_factory=lambda: smoke_val if _env_flag("SENTINEFIN_SMOKE") else full_val)


@dataclass
class DataConfig:
    """Phase 1: ingestion and windowing."""

    window: str = "M"  # monthly windows (pandas Period alias); e.g. "W" for weekly
    min_narrative_words: int = _profile(3, 10)


@dataclass
class EmbeddingConfig:
    """Phase 2: representation learning."""

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = _profile(16, 512)
    # Deterministic offline fallback used when the HF hub is unreachable (CI).
    hash_dim: int = _profile(128, 256)
    seed: int = 17


@dataclass
class MLPConfig:
    """Phase 2: neural baseline classifier."""

    hidden_dims: tuple[int, ...] = _profile((64, 32), (128, 64))
    epochs: int = _profile(2, 20)
    lr: float = 1e-3
    batch_size: int = _profile(64, 256)
    dropout: float = 0.2
    weight_decay: float = 1e-4


@dataclass
class DECConfig:
    """Phase 3: Deep Embedded Clustering."""

    n_clusters: int = _profile(8, 24)
    latent_dim: int = _profile(16, 32)
    hidden_dims: tuple[int, ...] = _profile((64, 32), (256, 128))
    pretrain_epochs: int = _profile(3, 60)
    finetune_iters: int = _profile(30, 8000)
    update_interval: int = _profile(5, 140)
    tol: float = 0.001
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    # Continual-learning controls for the rolling-window setting.
    warm_start: bool = True
    finetune_epochs_per_window: int = _profile(1, 4)
    replay_buffer_size: int = _profile(512, 4096)
    seed: int = 23


@dataclass
class DriftConfig:
    """Phase 4: LSTM Autoencoder drift/emergence scoring."""

    hidden_dim: int = _profile(32, 128)
    latent_dim: int = _profile(16, 48)
    epochs: int = _profile(60, 400)
    lr: float = 2e-3
    # Fraction of lowest-error training trajectories used to set the alert cut.
    quantile: float = 0.95
    margin: float = 0.05
    seed: int = 31


@dataclass
class BacktestConfig:
    """Phase 5: retrospective validation.

    Recognition dates are pre-registered before any backtest run and cite the
    sources listed in reports/backtest_cases.md.
    """

    cases: tuple[dict, ...] = (
        {
            "name": "buy-now-pay-later",
            # BNPL sub-issue values first appear in the CFPB public schema in 2023.
            "product": "Buy Now, Pay Later (BNPL)",
            "recognition_date": "2022-03-01",
            "recognition_basis": "CFPB BNPL market inquiry announced Dec 2021; "
            "first CFPB BNPL report Sep 2022",
        },
        {
            "name": "earned-wage-access",
            "product": "Earned Wage Access",
            "recognition_date": "2023-06-01",
            "recognition_basis": "CFPB EWA product-specific guidance activity in 2023",
        },
    )
    lead_time_months_required: int = 1


def pipeline_stages() -> list[str]:
    return ["ingest", "embed", "cluster", "drift"]
