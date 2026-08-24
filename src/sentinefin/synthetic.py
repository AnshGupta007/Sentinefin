"""Synthetic CFPB-like data used by the smoke profile and shared with tests.

The BNPL and crypto topics only appear in later months, giving the drift
stage an emerging pattern to flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TOPICS = [
    # (product, issue, vocabulary) — distinct vocabularies so DEC has signal.
    ("Credit reporting", "Wrong information on report",
     "credit report bureau dispute inaccurate score experian equifax section"),
    ("Debt collection", "Attempts to collect debt not owed",
     "collector debt owed harassment phone calls validation agency threaten"),
    ("Mortgage", "Trouble during payment process",
     "mortgage escrow lender servicer foreclosure payment modification house"),
    ("Buy Now, Pay Later (BNPL)", "Managing the loan or lease",
     "bnpl installment installmentpay checkout split purchase refund merchant"),
    ("Virtual currency", "Crypto asset not delivered",
     "crypto bitcoin wallet exchange withdrawal coins transfer platform"),
]

TEMPLATES = [
    "I contacted {v} about my account and they refused to help me at all.",
    "This company reported false {v} information and damaged my standing.",
    "They kept calling about a {v} matter even after I asked them to stop.",
    "My {v} payment was processed but never credited to the right place.",
    "After disputing, the {v} records were still wrong for months.",
]


def synthetic_complaints(
    n_per_topic: int = 40,
    start: str = "2021-01-01",
    end: str = "2022-12-31",
    seed: int = 7,
) -> pd.DataFrame:
    """Small CFPB-schema-shaped dataset with topic vocabularies and drift."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows = []
    for t_idx, (product, issue, vocab) in enumerate(TOPICS):
        words = vocab.split()
        for i in range(n_per_topic):
            # Emerging topics appear only in the second half of the range.
            day_pool = dates
            if product.startswith("Buy Now") or product == "Virtual currency":
                day_pool = dates[len(dates) // 2 :]
            day = day_pool[rng.integers(0, len(day_pool))]
            length = rng.integers(8, 30)
            body = " ".join(rng.choice(words + TEMPLATES[t_idx].split(), size=length))
            text = f"{TEMPLATES[t_idx]} {body}"
            rows.append({
                "complaint_id": 10_000 + t_idx * n_per_topic + i,
                "date_received": day.strftime("%m/%d/%Y"),
                "product": product,
                "issue": issue,
                "sub_issue": None,
                "consumer_complaint_narrative": text,
            })
    df = pd.DataFrame(rows)
    # Add some narrative-less rows to exercise the Phase 1 filter.
    no_narr = pd.DataFrame([
        {
            "complaint_id": 99_900 + j,
            "date_received": dates[rng.integers(0, len(dates))].strftime("%m/%d/%Y"),
            "product": "Other",
            "issue": "None",
            "sub_issue": None,
            "consumer_complaint_narrative": None,
        }
        for j in range(20)
    ])
    return pd.concat([df, no_narr], ignore_index=True)


def write_synthetic_raw(dest_dir: Path | None = None, n_per_topic: int = 40) -> Path:
    """Persist the synthetic fixture where ``load_raw`` will discover it."""
    from . import config as _cfg
    from .utils import ensure_dir

    dest_dir = dest_dir or _cfg.RAW_DATA_DIR
    ensure_dir(dest_dir)
    path = dest_dir / "complaints_smoke.csv"
    df = synthetic_complaints(n_per_topic=n_per_topic)
    df.to_csv(path, index=False)
    return path
