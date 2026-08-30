#!/usr/bin/env python3
"""Script to extract a high-quality, stratified small dataset from raw complaints.csv.

This creates a lightweight dataset (e.g., 20,000 complaints) suitable for training
deep learning models (Sentence Transformers, DEC clustering, LSTM-AE drift detection)
on resource-constrained machines without running out of memory.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_small_dataset")

DEFAULT_INPUT = Path("data/raw/complaints.csv")
DEFAULT_OUTPUT = Path("data/raw/complaints_small.csv")


def extract_small_dataset(
    input_path: Path | str = DEFAULT_INPUT,
    output_path: Path | str = DEFAULT_OUTPUT,
    target_size: int = 20_000,
    min_words: int = 10,
    seed: int = 42,
    chunk_size: int = 50_000,
    max_scan_chunks: int = 350,
    chunk_step: int = 10,
) -> Path:
    """Stream complaints.csv in chunks and extract a balanced, stratified small sample.

    Args:
        input_path: Path to raw complaints.csv (or .csv.zip).
        output_path: Destination path for the sampled CSV.
        target_size: Desired number of complaints in the output dataset.
        min_words: Minimum word count for the consumer narrative.
        seed: Random seed for reproducible sampling.
        chunk_size: Number of rows per streaming chunk.
        max_scan_chunks: Maximum chunks to scan across the dataset.
        chunk_step: Step size between sampled chunks to ensure wide temporal coverage.

    Returns:
        Path to the generated small dataset CSV.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    logger.info("Reading raw complaints from: %s", input_path)
    logger.info(
        "Target sample size: %d complaints (min narrative words: %d, seed: %d)",
        target_size,
        min_words,
        seed,
    )

    t0 = time.time()
    extracted_chunks = []
    total_narratives_collected = 0
    # Over-sample candidate pool to allow balanced stratification
    candidate_target = max(target_size * 3, 60_000)

    reader = pd.read_csv(
        input_path,
        chunksize=chunk_size,
        low_memory=False,
        dtype=str,  # preserve all columns as string for fidelity
    )

    for chunk_idx, chunk in enumerate(reader):
        if chunk_idx >= max_scan_chunks:
            logger.info("Reached maximum scan limit of %d chunks.", max_scan_chunks)
            break

        # Step through chunks to span different time periods in the file
        if chunk_idx % chunk_step != 0:
            continue

        # Find narrative column (handling potential casing differences)
        narr_col = None
        for col in chunk.columns:
            if col.strip().lower().replace(" ", "_").replace("-", "_") == "consumer_complaint_narrative":
                narr_col = col
                break

        if narr_col is None:
            continue

        # Filter for rows with non-empty narratives
        has_narrative = (
            chunk[narr_col].notna()
            & (chunk[narr_col].astype(str).str.strip() != "")
            & (chunk[narr_col].astype(str).str.strip().str.lower() != "nan")
        )
        subset = chunk.loc[has_narrative].copy()

        if len(subset) == 0:
            continue

        # Filter by minimum word count
        word_counts = subset[narr_col].astype(str).str.count(r"\S+")
        subset = subset.loc[word_counts >= min_words]

        if len(subset) > 0:
            extracted_chunks.append(subset)
            total_narratives_collected += len(subset)
            logger.info(
                "Chunk %d: collected %d valid narratives (total candidates: %d / %d)",
                chunk_idx,
                len(subset),
                total_narratives_collected,
                candidate_target,
            )

        if total_narratives_collected >= candidate_target:
            logger.info("Candidate pool target reached. Proceeding to stratification.")
            break

    if not extracted_chunks:
        raise ValueError("No valid complaint narratives found in input file.")

    candidates_df = pd.concat(extracted_chunks, ignore_index=True)
    logger.info(
        "Collected %d candidate complaints in %.1fs.",
        len(candidates_df),
        time.time() - t0,
    )

    # Normalize column names for internal processing
    date_col = next(
        (c for c in candidates_df.columns if "date" in c.lower() and "received" in c.lower()),
        "Date received",
    )
    product_col = next(
        (c for c in candidates_df.columns if c.strip().lower() == "product"),
        "Product",
    )

    # Parse dates and determine time periods
    candidates_df["_dt"] = pd.to_datetime(candidates_df[date_col], errors="coerce")
    candidates_df = candidates_df.dropna(subset=["_dt"]).copy()
    candidates_df["_period"] = candidates_df["_dt"].dt.to_period("M").astype(str)

    # Clean product column
    candidates_df["_prod"] = candidates_df[product_col].fillna("Unknown").astype(str).str.strip()

    # Stratified sampling across (period, product)
    total_available = len(candidates_df)
    if total_available <= target_size:
        logger.warning(
            "Available valid complaints (%d) <= target_size (%d). Using all available.",
            total_available,
            target_size,
        )
        final_sample = candidates_df
    else:
        # Group by period and product to sample proportionally with a cap to maintain diversity
        groups = candidates_df.groupby(["_period", "_prod"])
        n_groups = len(groups)
        base_per_group = max(1, target_size // n_groups)

        # Allocate samples per group
        sample_indices = []
        for _, grp in groups:
            n_select = min(len(grp), max(base_per_group, int(len(grp) * (target_size / total_available))))
            chosen = rng.choice(grp.index, size=n_select, replace=False)
            sample_indices.extend(chosen)

        # If we have more or less than target_size, adjust
        sample_indices = list(set(sample_indices))
        if len(sample_indices) > target_size:
            sample_indices = rng.choice(sample_indices, size=target_size, replace=False).tolist()
        elif len(sample_indices) < target_size:
            remaining = list(set(candidates_df.index) - set(sample_indices))
            needed = target_size - len(sample_indices)
            additional = rng.choice(remaining, size=min(needed, len(remaining)), replace=False).tolist()
            sample_indices.extend(additional)

        final_sample = candidates_df.loc[sample_indices].copy()

    # Remove temporary helper columns
    final_sample = final_sample.drop(columns=["_dt", "_period", "_prod"])

    # Sort chronologically by date received if available
    if date_col in final_sample.columns:
        final_sample["_dt_sort"] = pd.to_datetime(final_sample[date_col], errors="coerce")
        final_sample = final_sample.sort_values("_dt_sort").drop(columns=["_dt_sort"])

    # Save to CSV
    final_sample.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info("Successfully created small dataset: %s", output_path)
    logger.info("Final row count: %d complaints", len(final_sample))
    logger.info("File size: %.2f MB", file_size_mb)
    logger.info(
        "Date span: %s to %s",
        final_sample[date_col].min(),
        final_sample[date_col].max(),
    )
    logger.info("Top 5 products in sample:\n%s", final_sample[product_col].value_counts().head(5).to_string())

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a small, balanced dataset from CFPB complaints.csv for model training."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input raw complaints CSV (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output sampled CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=20_000,
        help="Number of complaints in sample (default: 20000)",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=10,
        help="Minimum narrative word count (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=10,
        help="Chunk step size for scanning (default: 10)",
    )

    args = parser.parse_args()

    try:
        extract_small_dataset(
            input_path=args.input,
            output_path=args.output,
            target_size=args.n_samples,
            min_words=args.min_words,
            seed=args.seed,
            chunk_step=args.step,
        )
        return 0
    except Exception as exc:
        logger.error("Failed to create small dataset: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
