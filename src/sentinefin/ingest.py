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


def load_raw(path: Path | None = None, **read_kwargs: object) -> pd.DataFrame:
    """Load the raw CFPB csv (zip or extracted), or a synthetic fixture."""
    if path is None:
        candidates = sorted(_cfg.RAW_DATA_DIR.glob("complaints*.csv*"))
        if not candidates:
            raise FileNotFoundError(
                f"No raw CFPB file under {_cfg.RAW_DATA_DIR}. Run `sentinefin ingest` first."
            )
        path = candidates[0]
    kwargs: dict[str, object] = {"low_memory": False}
    kwargs.update(read_kwargs)
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".csv"))
            df = pd.read_csv(zf.open(name), **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
    return df


def filter_with_narratives(df: pd.DataFrame, min_words: int) -> pd.DataFrame:
    """Keep rows with narratives of at least ``min_words`` words; add stats."""
    total = len(df)
    has_narrative = df[NARRATIVE_COL].notna() & (df[NARRATIVE_COL].astype(str).str.strip() != "")
    out = df.loc[has_narrative].copy()
    pct = 100.0 * has_narrative.mean()
    logger.info("%.2f%% of %d complaints carry a narrative", pct, total)
    out.attrs["narrative_rate"] = pct
    out["narrative_word_count"] = out[NARRATIVE_COL].astype(str).str.split().str.len()
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
    top_products.rename_axis(PRODUCT_COL).to_frame("count").to_csv(
        outputs_dir / "top_products.csv"
    )
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
