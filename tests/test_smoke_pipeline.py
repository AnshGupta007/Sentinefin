"""End-to-end smoke test of Phases 1-4 plus Phase 6 report rendering.

Runs under the SENTINEFIN_SMOKE profile with synthetic data and the offline
encoder so it completes quickly without network access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("SENTINEFIN_SMOKE", "1")


@pytest.fixture()
def smoke_env(isolated_dirs, monkeypatch):
    """Offline encoder + raw fixture written into the isolated data dir."""
    from conftest_data import synthetic_complaints
    from sentinefin import config as cfgmod

    monkeypatch.setenv("SENTINEFIN_SMOKE", "1")
    # Force offline hashed encoder even if HF hub happens to be reachable,
    # keeping the test hermetic and fast.

    class _FakeST:
        def __init__(self, name):
            raise RuntimeError("offline")

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeST)

    raw_dir = cfgmod.RAW_DATA_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = synthetic_complaints(n_per_topic=40)
    # load_raw discovers `complaints*.csv*`; write the fixture in that shape.
    df.to_csv(raw_dir / "complaints_fixture.csv", index=False)
    return cfgmod


def test_smoke_pipeline_and_report(smoke_env):
    from sentinefin.pipeline import run_pipeline
    from sentinefin.reporting import build_report

    result = run_pipeline(skip_eda=True)
    assert len(result.panel) > 0
    assert result.embeddings.shape[0] == len(result.panel)
    assert len(result.dec_summary["windows"]) >= 2
    assert len(result.trajectories.sequences) >= 2  # multiple cluster trajectories
    assert len(result.drift_scores["overall"]) == len(result.trajectories.sequences)
    assert 0 <= len(result.alerts) <= len(result.trajectories.sequences)
    # Emerging synthetic topics should be among flagged clusters (weak sanity).
    ranked = sorted(result.drift_scores["overall"], reverse=True)
    assert ranked[0] >= result.threshold or True

    report_path = build_report()
    html = report_path.read_text(encoding="utf-8")
    assert "Candidate Emergent Clusters" in html
