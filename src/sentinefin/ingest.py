"""Phase 1: CFPB data ingestion, filtering, time-window paneling, and EDA."""

from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config as _cfg
from .config import CFPB_URL, DataConfig
from .utils import ensure_dir

logger = logging.getLogger(__name__)

NARRATIVE_COL = "consumer_complaint_narrative"
DATE_COL = "date_received"
PRODUCT_COL = "product"
ISSUE_COL = "issue"
SUB_ISSUE_COL = "sub_issue"


def download_raw(dest_dir: Path | None = None, url: str = CFPB_URL) -> Path:
    """Download the CFPB complaints zip (or reuse a cached copy)."""
    dest_dir = dest_dir or _cfg.RAW_DATA_DIR
    ensure_dir(dest_dir)
    dest = dest_dir / "complaints.csv.zip"
    if dest.exists():
        logger.info("Reusing cached download at %s", dest)
        return dest
    logger.info("Downloading %s ...", url)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - fixed public URL
    return dest


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw CSV headers to snake_case (e.g. 'Date received' -> 'date_received')."""
    df.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


def load_raw(
    path: Path | None = None, chunksize: int = 100_000, **read_kwargs: object
) -> pd.DataFrame:
    """Load the raw CFPB csv (zip or extracted), streaming in chunks if large."""
    if path is None:
        small_candidate = _cfg.RAW_DATA_DIR / "complaints_small.csv"
        if small_candidate.exists():
            path = small_candidate
            logger.info("Auto-selected small dataset: %s", path)
        else:
            candidates = sorted(_cfg.RAW_DATA_DIR.glob("complaints*.csv*"))
            if not candidates:
                raise FileNotFoundError(
                    f"No raw CFPB file under {_cfg.RAW_DATA_DIR}. Run `sentinefin ingest` first."
                )
            path = candidates[0]

    if path.stat().st_size < 50_000_000:
        kwargs: dict[str, object] = {"low_memory": False}
        kwargs.update(read_kwargs)
        if str(path).endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                name = next(n for n in zf.namelist() if n.endswith(".csv"))
                df = pd.read_csv(zf.open(name), **kwargs)
        else:
            df = pd.read_csv(path, **kwargs)
        return _normalize_cols(df)

    logger.info("Loading large raw file %s in chunks of %d ...", path, chunksize)
    chunks = []
    total_raw = 0
    total_kept = 0

    if str(path).endswith(".zip"):
        zf = zipfile.ZipFile(path)
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        src = zf.open(name)
    else:
        src = str(path)

    for chunk in pd.read_csv(src, chunksize=chunksize, low_memory=False, **read_kwargs):
        total_raw += len(chunk)
        chunk = _normalize_cols(chunk)
        if NARRATIVE_COL in chunk.columns:
            has_narrative = chunk[NARRATIVE_COL].notna() & (
                chunk[NARRATIVE_COL].astype(str).str.strip() != ""
            )
            chunk = chunk[has_narrative]
        total_kept += len(chunk)
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Streamed %d raw rows -> kept %d rows with narratives (%.2f%%)",
        total_raw,
        total_kept,
        100.0 * total_kept / max(1, total_raw),
    )
    return df


def filter_with_narratives(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    """Keep rows with narratives of at least ``min_words`` words; add stats."""
    total = len(df)
    has_narrative = df[NARRATIVE_COL].notna() & (df[NARRATIVE_COL].astype(str).str.strip() != "")
    out = df.loc[has_narrative].copy()
    pct = 100.0 * has_narrative.mean()
    logger.info("%.2f%% of %d complaints carry a narrative", pct, total)
    out.attrs["narrative_rate"] = pct
    out["narrative_word_count"] = out[NARRATIVE_COL].astype(str).str.count(r"\S+")
    before = len(out)
    out = out[out["narrative_word_count"] >= min_words]
    dropped = before - len(out)
    if dropped:
        logger.info("Dropped %d narratives shorter than %d words", dropped, min_words)
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.dropna(subset=[DATE_COL])
    return out.sort_values(DATE_COL).reset_index(drop=True)


def assign_windows(df: pd.DataFrame, window: str) -> pd.DataFrame:
    """Attach a window id column like ``2021-03`` based on the date column."""
    periods = df[DATE_COL].dt.to_period(window)
    df = df.copy()
    df["window_id"] = periods.astype(str)
    return df


def build_panel(
    df: pd.DataFrame,
    config: DataConfig | None = None,
    processed_dir: Path | None = None,
) -> pd.DataFrame:
    """Full Phase-1 transform: filter, clean, window, save parquet + notes."""
    cfg = config or DataConfig()
    processed_dir = processed_dir or _cfg.PROCESSED_DATA_DIR
    out = filter_with_narratives(df, min_words=cfg.min_narrative_words)
    out = assign_windows(out, cfg.window)
    keep = [
        "complaint_id",
        DATE_COL,
        "window_id",
        PRODUCT_COL,
        ISSUE_COL,
        SUB_ISSUE_COL,
        NARRATIVE_COL,
        "narrative_word_count",
    ]
    keep = [c for c in keep if c in out.columns]
    panel = out[keep]
    ensure_dir(processed_dir)
    target = processed_dir / "panel.parquet"
    panel.to_parquet(target, index=False)
    logger.info("Wrote panel with %d complaints to %s", len(panel), target)
    return panel


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def run_eda(panel: pd.DataFrame, outputs_dir: Path | None = None) -> list[Path]:
    """Save summary plots + CSVs describing the complaint panel."""
    outputs_dir = outputs_dir or _cfg.OUTPUTS_DIR
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_dir(outputs_dir)
    paths: list[Path] = []

    # Volume over time
    vol = panel.groupby("window_id").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    vol.plot(ax=ax)
    ax.set_title("Complaint volume per window")
    ax.set_ylabel("complaints")
    fig.tight_layout()
    p = outputs_dir / "eda_volume_over_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Narrative length distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    panel["narrative_word_count"].clip(upper=600).hist(bins=50, ax=ax)
    ax.set_title("Narrative length distribution")
    ax.set_xlabel("words")
    fig.tight_layout()
    p = outputs_dir / "eda_narrative_length.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Top products/issues
    top_products = panel[PRODUCT_COL].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_products.iloc[::-1].plot.barh(ax=ax)
    ax.set_title("Top products")
    fig.tight_layout()
    p = outputs_dir / "eda_top_products.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    # Summary tables
    vol.rename("count").to_csv(outputs_dir / "volume_by_window.csv")
    top_products.rename_axis(PRODUCT_COL).to_frame("count").to_csv(outputs_dir / "top_products.csv")
    return paths


def summarize_panel(panel: pd.DataFrame) -> dict:
    windows = sorted(panel["window_id"].unique())
    return {
        "n_complaints": int(len(panel)),
        "n_windows": len(windows),
        "first_window": windows[0] if windows else None,
        "last_window": windows[-1] if windows else None,
        "n_products": int(panel[PRODUCT_COL].nunique()),
        "median_narrative_words": float(panel["narrative_word_count"].median())
        if len(panel)
        else 0.0,
    }


def sample_raw_dataset(
    input_path: Path | str | None = None,
    output_path: Path | str | None = None,
    target_size: int = 20_000,
    min_words: int = 10,
    seed: int = 42,
    chunk_size: int = 50_000,
    max_scan_chunks: int = 350,
    chunk_step: int = 10,
) -> Path:
    """Stream raw CFPB CSV and create a balanced, stratified small sample.

    Extracts complaints with substantial narratives, distributed across monthly
    windows and product categories to enable training on low-compute infrastructure.
    """
    import numpy as np

    if input_path is None:
        candidates = sorted(_cfg.RAW_DATA_DIR.glob("complaints*.csv*"))
        candidates = [
            c
            for c in candidates
            if "small" not in c.name and "smoke" not in c.name and "fixture" not in c.name
        ]
        if not candidates:
            raise FileNotFoundError(f"No source complaints CSV found under {_cfg.RAW_DATA_DIR}")
        input_path = candidates[0]

    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else _cfg.RAW_DATA_DIR / "complaints_small.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    logger.info("Sampling %d complaints from %s -> %s", target_size, input_path, output_path)

    extracted_chunks = []
    total_narratives = 0
    candidate_target = max(target_size * 3, 60_000)

    reader = pd.read_csv(input_path, chunksize=chunk_size, low_memory=False, dtype=str)

    for chunk_idx, chunk in enumerate(reader):
        if chunk_idx >= max_scan_chunks:
            break
        if chunk_idx % chunk_step != 0:
            continue

        narr_col = None
        for col in chunk.columns:
            if col.strip().lower().replace(" ", "_").replace("-", "_") == NARRATIVE_COL:
                narr_col = col
                break

        if narr_col is None:
            continue

        has_narrative = (
            chunk[narr_col].notna()
            & (chunk[narr_col].astype(str).str.strip() != "")
            & (chunk[narr_col].astype(str).str.strip().str.lower() != "nan")
        )
        subset = chunk.loc[has_narrative].copy()
        if len(subset) == 0:
            continue

        word_counts = subset[narr_col].astype(str).str.count(r"\S+")
        subset = subset.loc[word_counts >= min_words]

        if len(subset) > 0:
            extracted_chunks.append(subset)
            total_narratives += len(subset)

        if total_narratives >= candidate_target:
            break

    if not extracted_chunks:
        raise ValueError(f"No valid complaint narratives found in {input_path}")

    candidates_df = pd.concat(extracted_chunks, ignore_index=True)
    date_col = next(
        (c for c in candidates_df.columns if "date" in c.lower() and "received" in c.lower()),
        "Date received",
    )
    product_col = next(
        (c for c in candidates_df.columns if c.strip().lower() == "product"),
        "Product",
    )

    candidates_df["_dt"] = pd.to_datetime(candidates_df[date_col], errors="coerce")
    candidates_df = candidates_df.dropna(subset=["_dt"]).copy()
    candidates_df["_period"] = candidates_df["_dt"].dt.to_period("M").astype(str)
    candidates_df["_prod"] = candidates_df[product_col].fillna("Unknown").astype(str).str.strip()

    total_available = len(candidates_df)
    if total_available <= target_size:
        final_sample = candidates_df
    else:
        groups = candidates_df.groupby(["_period", "_prod"])
        base_per_group = max(1, target_size // max(1, len(groups)))
        sample_indices = []
        for _, grp in groups:
            n_select = min(
                len(grp), max(base_per_group, int(len(grp) * (target_size / total_available)))
            )
            chosen = rng.choice(grp.index, size=n_select, replace=False)
            sample_indices.extend(chosen)

        sample_indices = list(set(sample_indices))
        if len(sample_indices) > target_size:
            sample_indices = rng.choice(sample_indices, size=target_size, replace=False).tolist()
        elif len(sample_indices) < target_size:
            remaining = list(set(candidates_df.index) - set(sample_indices))
            needed = target_size - len(sample_indices)
            additional = rng.choice(
                remaining, size=min(needed, len(remaining)), replace=False
            ).tolist()
            sample_indices.extend(additional)

        final_sample = candidates_df.loc[sample_indices].copy()

    final_sample = final_sample.drop(columns=["_dt", "_period", "_prod"])
    if date_col in final_sample.columns:
        final_sample["_dt_sort"] = pd.to_datetime(final_sample[date_col], errors="coerce")
        final_sample = final_sample.sort_values("_dt_sort").drop(columns=["_dt_sort"])

    final_sample.to_csv(output_path, index=False)
    logger.info(
        "Saved %d sampled complaints (%.2f MB) to %s",
        len(final_sample),
        output_path.stat().st_size / (1024 * 1024),
        output_path,
    )
    return output_path
