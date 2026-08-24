"""Shared fixtures: redirect project dirs to tmp and provide synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest_data import synthetic_complaints  # noqa: E402


@pytest.fixture()
def isolated_dirs(tmp_path, monkeypatch):
    """Point all SentinelFin artifact dirs at a temp dir for test isolation."""
    from sentinefin import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "RAW_DATA_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", tmp_path / "data" / "processed")
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    return tmp_path


@pytest.fixture()
def synthetic_df():
    return synthetic_complaints(n_per_topic=30)
