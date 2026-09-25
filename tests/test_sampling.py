"""Tests for raw complaints sampling and small dataset creation."""

from __future__ import annotations

import pandas as pd
import pytest

from sentinefin.ingest import sample_raw_dataset


@pytest.fixture
def raw_complaints_csv(tmp_path):
    """Create a mock raw complaints CSV file."""
    rows = []
    products = ["Credit reporting", "Debt collection", "Mortgage", "Credit card"]
    dates = ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05"]

    for i in range(100):
        prod = products[i % len(products)]
        dt = dates[i % len(dates)]
        # Some rows with short or empty narratives
        if i % 10 == 0:
            narr = ""
        elif i % 10 == 1:
            narr = "Too short"
        else:
            narr = f"This is a valid consumer complaint narrative about {prod} for customer {i} with sufficient word count."

        rows.append(
            {
                "Date received": dt,
                "Product": prod,
                "Sub-product": "General",
                "Issue": "Billing dispute",
                "Sub-issue": "None",
                "Consumer complaint narrative": narr,
                "Company": f"Bank {i % 5}",
                "State": "CA",
                "ZIP code": "90210",
                "Complaint ID": str(100000 + i),
            }
        )

    csv_path = tmp_path / "test_raw_complaints.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_sample_raw_dataset_filtering_and_size(raw_complaints_csv, tmp_path):
    out_path = tmp_path / "sampled.csv"
    target_size = 20

    res_path = sample_raw_dataset(
        input_path=raw_complaints_csv,
        output_path=out_path,
        target_size=target_size,
        min_words=5,
        seed=42,
        chunk_size=25,
        chunk_step=1,
    )

    assert res_path.exists()
    df = pd.read_csv(res_path)
    assert len(df) == target_size
    assert "Date received" in df.columns
    assert "Product" in df.columns
    assert "Consumer complaint narrative" in df.columns
    assert df["Consumer complaint narrative"].notna().all()
    # Ensure min words condition met
    word_counts = df["Consumer complaint narrative"].astype(str).str.count(r"\S+")
    assert (word_counts >= 5).all()


def test_sample_raw_dataset_reproducibility(raw_complaints_csv, tmp_path):
    out1 = tmp_path / "s1.csv"
    out2 = tmp_path / "s2.csv"

    sample_raw_dataset(
        input_path=raw_complaints_csv,
        output_path=out1,
        target_size=15,
        min_words=5,
        seed=123,
        chunk_size=30,
        chunk_step=1,
    )
    sample_raw_dataset(
        input_path=raw_complaints_csv,
        output_path=out2,
        target_size=15,
        min_words=5,
        seed=123,
        chunk_size=30,
        chunk_step=1,
    )

    df1 = pd.read_csv(out1)
    df2 = pd.read_csv(out2)
    pd.testing.assert_frame_equal(df1, df2)
