"""Phase 1 tests: filtering, windowing, panel building."""

from __future__ import annotations

import pandas as pd

from conftest_data import synthetic_complaints
from sentinefin.config import DataConfig
from sentinefin.ingest import (
    NARRATIVE_COL,
    assign_windows,
    build_panel,
    filter_with_narratives,
    summarize_panel,
)


def test_filter_removes_missing_narratives():
    df = synthetic_complaints(n_per_topic=10)
    filtered = filter_with_narratives(df, min_words=3)
    assert filtered[NARRATIVE_COL].notna().all()
    assert len(filtered) == 50  # 5 topics x 10; the 20 null-narrative rows are gone


def test_windows_are_monthly_and_configurable():
    df = synthetic_complaints(n_per_topic=10)
    filtered = filter_with_narratives(df, min_words=3)
    monthly = assign_windows(filtered, "M")
    assert monthly["window_id"].str.match(r"^\d{4}-\d{2}$").all()
    weekly = assign_windows(filtered, "W")
    assert weekly["window_id"].nunique() >= monthly["window_id"].nunique()


def test_build_panel_writes_parquet(tmp_path):
    df = synthetic_complaints(n_per_topic=15)
    processed = tmp_path / "processed"
    panel = build_panel(df, config=DataConfig(), processed_dir=processed)
    assert (processed / "panel.parquet").exists()
    summary = summarize_panel(panel)
    assert summary["n_complaints"] == len(panel) == 75
    assert summary["n_windows"] >= 2
    assert pd.api.types.is_datetime64_any_dtype(panel["date_received"])
