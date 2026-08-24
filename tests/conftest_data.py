"""Synthetic CFPB-like fixture shared by tests and the smoke pipeline."""

from __future__ import annotations

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
    """Small CFPB-schema-shaped dataset with topic vocabularies and drift.

    The BNPL and crypto topics only appear in later months, simulating an
    emerging harm pattern for the drift stage to flag.
    """
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
