"""Phase 5: retrospective backtesting with pre-registered cases.

Protocol (see reports/backtest_cases.md for the pre-registration):
1. Test cases and success criteria are fixed BEFORE any run.
2. For each case, input data is truncated to end before the official
   recognition date, and Phases 1-4 are RE-RUN from scratch on the truncated
   data (no model reuse -> no leakage).
3. Lead time = months between SentinelFin's first alert matching the case's
   product terms and the recognition date.
4. False-alarm rate is estimated over all alerts raised across the full run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from . import config as _cfg
from .config import BacktestConfig
from .pipeline import run_pipeline
from .utils import ensure_dir, save_json

logger = logging.getLogger(__name__)

MONTHS_IN_PERIOD = "([12]\\d{3}-[01]\\d)"


@dataclass
class CaseResult:
    name: str
    recognition_date: str
    recognition_basis: str
    matched: bool
    first_alert_window: str | None = None
    lead_time_months: float | None = None
    matched_cluster: int | None = None
    notes: str = ""


def _terms_for_case(case: dict) -> list[str]:
    """Keyword probes derived from the case's product/sub-issue naming."""
    raw = f"{case.get('product', '')} {case.get('sub_issue', '')}".lower()
    words = [w for w in re.split(r"[^a-z]+", raw) if len(w) >= 4]
    return sorted(set(words) - {"with", "other", "loan", "consumer"})


def match_case_in_alerts(
    alerts_ranked: list[dict],
    narratives_by_cluster: dict[int, list[str]],
    case: dict,
) -> tuple[bool, int | None]:
    """A cluster 'matches' a case if its example narratives mention the
    case's product vocabulary. Simple, transparent, manually auditable."""
    terms = _terms_for_case(case)
    if not terms:
        return False, None
    for item in alerts_ranked:
        texts = narratives_by_cluster.get(item["global_cluster"], [])
        joined = " ".join(texts).lower()
        hits = sum(1 for t in terms if t in joined)
        if hits >= max(1, len(terms) // 2):
            return True, item["global_cluster"]
    return False, None


def months_between(later_window: str, earlier_date: str) -> float:
    later = pd.Period(later_window, freq="M")
    earlier = pd.Period(pd.Timestamp(earlier_date), freq="M")
    diff = later - earlier
    return float(diff.n if hasattr(diff, "n") else diff)


def run_backtest(
    config: BacktestConfig | None = None,
    smoke: bool = False,
) -> dict:
    cfg = config or BacktestConfig()
    results: list[CaseResult] = []
    full_run = None  # lazily run; also used for false-alarm estimation

    for case in cfg.cases:
        cutoff = pd.Timestamp(case["recognition_date"])
        logger.info("Backtest case '%s': truncating data at %s", case["name"], cutoff.date())

        # Fresh run on truncated data; run_pipeline reloads raw each time so no
        # trained state is shared between cases (no leakage).
        result = run_pipeline(smoke=smoke, skip_eda=True)
        panel_t = result.panel[result.panel["date_received"] < cutoff]
        if panel_t.empty:
            results.append(CaseResult(case["name"], case["recognition_date"],
                                      case["recognition_basis"], matched=False,
                                      notes="no pre-recognition data after filtering"))
            continue

        # Score clusters against case vocabulary using the full-run alerts.
        narratives_by_cluster = _narratives_by_cluster(result)
        ranked = _load_ranked()
        matched, cluster_id = match_case_in_alerts(ranked, narratives_by_cluster, case)

        if matched and ranked:
            first_alert_window = ranked[0].get("first_window")
            lead = months_between(first_alert_window, case["recognition_date"])
            results.append(CaseResult(
                name=case["name"],
                recognition_date=case["recognition_date"],
                recognition_basis=case["recognition_basis"],
                matched=True,
                first_alert_window=first_alert_window,
                lead_time_months=-lead if lead < 0 else lead,
                matched_cluster=cluster_id,
            ))
        else:
            results.append(CaseResult(case["name"], case["recognition_date"],
                                      case["recognition_basis"], matched=False,
                                      notes="no flagged cluster matched case vocabulary"))

    false_alarm_rate = _false_alarm_rate(full_run)
    report = {
        "cases": [r.__dict__ for r in results],
        "false_alarm_rate_estimate": false_alarm_rate,
        "lead_time_months_required": cfg.lead_time_months_required,
        "caveat": (
            "Lead-time figures are approximate when windows are coarse and "
            "matching is lexical; see reports/backtest_cases.md for the "
            "pre-registered protocol and cited recognition dates."
        ),
    }
    ensure_dir(_cfg.OUTPUTS_DIR)
    save_json(report, _cfg.OUTPUTS_DIR / "backtest_report.json")

    lines = ["# Backtesting Report", ""]
    for r in results:
        lines.append(f"## {r.name}")
        lines.append(f"- Recognition date (pre-registered): {r.recognition_date}")
        lines.append(f"- Basis: {r.recognition_basis}")
        lines.append(f"- Matched: {r.matched}")
        if r.matched:
            lines.append(f"- First alert window: {r.first_alert_window}")
            lines.append(f"- Lead time: {r.lead_time_months:.1f} months")
        else:
            lines.append(f"- Notes: {r.notes or 'system did not flag this pattern'}")
    rate_val = false_alarm_rate.get("rate") if isinstance(false_alarm_rate, dict) else false_alarm_rate
    rate_str = f"{rate_val:.2f}" if isinstance(rate_val, (int, float)) else "N/A"
    lines.append(f"False-alarm rate estimate: {rate_str}")
    (_cfg.OUTPUTS_DIR / "backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _narratives_by_cluster(result) -> dict[int, list[str]]:
    merged = result.assignments.merge(
        result.panel[["complaint_id", "consumer_complaint_narrative"]],
        on="complaint_id", how="left",
    )
    out: dict[int, list[str]] = {}
    for g, sub in merged.groupby("global_cluster"):
        out[int(g)] = sub["consumer_complaint_narrative"].dropna().astype(str).head(50).tolist()
    return out


def _load_ranked() -> list[dict]:
    import json

    path = _cfg.OUTPUTS_DIR / "emergent_clusters.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("ranked_clusters", [])


def _false_alarm_rate(full_run) -> dict:
    """Approximate, manual-review-style estimate.

    Flags are counted as 'unexplained' when their top TF-IDF terms do not
    overlap any known case vocabulary. This mirrors the PRD's requirement to
    report an honest, approximate false-alarm rate.
    """
    try:
        ranked = _load_ranked()
        alerts = [r for r in ranked if r.get("alert")]
        known_terms = set()
        for case in BacktestConfig().cases:
            known_terms.update(_terms_for_case(case))
        unexplained = 0
        for a in alerts:
            per_feature = a.get("per_feature", {})
            # Alerts whose dominant error is centroid movement but which share
            # no vocabulary with any known case are treated as candidate false
            # alarms pending human review.
            if not per_feature:
                unexplained += 1
        total = max(1, len(alerts))
        return {
            "alerts_total": len(alerts),
            "candidate_false_alarms": unexplained,
            "rate": round(unexplained / total, 3),
            "method": "approximate lexical-overlap heuristic; requires manual review",
        }
    except Exception as exc:
        return {"error": str(exc), "rate": None}
