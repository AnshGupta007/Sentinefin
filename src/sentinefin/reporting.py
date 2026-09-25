"""Phase 6: static HTML report of currently emergent clusters.

Deliberately minimal per the PRD: a single self-contained HTML page listing
ranked candidate emergent clusters with score breakdowns and readable example
narratives. Falls back to top-term labeling when no LLM key is available.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

import pandas as pd

from . import config as _cfg
from .utils import ensure_dir

logger = logging.getLogger(__name__)


def top_terms(texts: list[str], k: int = 6) -> list[str]:
    """Fallback cluster labeling: top TF-IDF terms."""
    if not texts:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(stop_words="english", max_features=2000)
        matrix = vec.fit_transform(texts)
        scores = matrix.sum(axis=0).A1
        terms = vec.get_feature_names_out()
        order = scores.argsort()[::-1][:k]
        return [terms[i] for i in order]
    except Exception:
        words: dict[str, int] = {}
        for t in texts:
            for w in str(t).lower().split():
                if len(w) > 4:
                    words[w] = words.get(w, 0) + 1
        return [w for w, _ in sorted(words.items(), key=lambda kv: -kv[1])[:k]]


def optional_llm_label(texts: list[str]) -> str | None:
    """One-line LLM summary if OPENAI_API_KEY exists; None otherwise."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        sample = " ".join(texts[:10])[:4000]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the common complaint pattern in one short line.",
                },
                {"role": "user", "content": sample},
            ],
            max_tokens=60,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception as exc:
        logger.warning("LLM labeling unavailable (%s); using TF-IDF terms.", exc)
        return None


def build_report(
    emergent_json: Path | None = None,
    panel_path: Path | None = None,
    assignments_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Render the static HTML report; returns the written path."""
    emergent_json = emergent_json or _cfg.OUTPUTS_DIR / "emergent_clusters.json"
    panel_path = panel_path or _cfg.PROCESSED_DATA_DIR / "panel.parquet"
    assignments_path = assignments_path or _cfg.PROCESSED_DATA_DIR / "dec_assignments.parquet"
    out_dir = out_dir or _cfg.REPORTS_DIR

    data = json.loads(emergent_json.read_text(encoding="utf-8"))
    ranked = data.get("ranked_clusters", [])
    threshold = data.get("threshold")

    panel = pd.read_parquet(panel_path) if panel_path.exists() else None
    assigns = pd.read_parquet(assignments_path) if assignments_path.exists() else None

    sections = []
    for item in ranked:
        g = item["global_cluster"]
        narratives: list[str] = []
        if panel is not None and assigns is not None:
            ids = assigns.loc[assigns["global_cluster"] == g, "complaint_id"]
            narratives = (
                panel.loc[panel["complaint_id"].isin(ids), "consumer_complaint_narrative"]
                .dropna()
                .astype(str)
                .head(5)
                .tolist()
            )
        label = optional_llm_label(narratives)
        label_text = (
            html.escape(label)
            if label
            else ", ".join(html.escape(t) for t in top_terms(narratives))
            or "(no narratives available)"
        )
        badge = (
            '<span style="color:#b00;font-weight:bold;">ALERT</span>' if item.get("alert") else ""
        )
        examples = "".join(
            f"<li>{html.escape(n[:600] + ('...' if len(n) > 600 else ''))}</li>" for n in narratives
        )
        feats = "".join(
            f"<code>{html.escape(k)}</code>: {v:.4f} &nbsp; "
            for k, v in item.get("per_feature", {}).items()
        )
        sections.append(f"""
<h2>Cluster {g} {badge}</h2>
<p><b>Error:</b> {item["error"]:.4f} &nbsp; <b>First seen:</b> {item["first_window"]}</p>
<p><b>Label:</b> {label_text}</p>
<p>{feats}</p>
<ul>{examples}</ul>
""")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>SentinelFin - Emergent Clusters</title>
<style>body{{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;}}
h1{{border-bottom:2px solid #333}} code{{background:#f4f4f4;padding:1px 3px}}</style>
</head><body>
<h1>SentinelFin - Candidate Emergent Clusters</h1>
<p>Alert threshold (reconstruction error): {threshold}<br>
Generated from the most recent pipeline run. Scores are LSTM-AE reconstruction
errors with interpretable sub-components.</p>
{"".join(sections) or "<p>No clusters scored yet.</p>"}
</body></html>"""
    ensure_dir(out_dir)
    out = out_dir / "sentinefin_report.html"
    out.write_text(doc, encoding="utf-8")
    logger.info("Report written to %s", out)
    return out
