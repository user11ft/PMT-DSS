"""
nlp_recommendation.py
----------------------
The "flexible NLP recommendation layer" on top of gap_recommendation.py's
knowledge base.

PROBLEM this solves: previously, every gap for the same indicator got the
exact same reason/action sentence, whether it just slipped 1 point below
target for the first time or has been critically failing for 8 months
straight. Two different gaps should not get the same recommendation.

HOW: for every (Department, Indicator, Year, Month) gap row, we:
  1. Compute CONTEXT FEATURES from the time series itself:
       - Trend (Improving / Worsening / Stable) vs. last month
       - Volatility over the trailing 6 months
       - Consecutive-months-in-gap streak (how long this has persisted)
  2. Run a lightweight NLG (natural-language generation) engine that
     assembles the final sentence from a pool of paraphrase fragments,
     selected deterministically from the row's own context (severity +
     trend + streak), and slot-fills the real numbers (rate, target,
     streak length) into the sentence.
  3. Produce an Urgency_Score (0-100) that blends severity, trend
     direction and persistence -- used to re-rank/prioritize the table
     so the *worst compounding* problems surface first, not just
     whatever sorts alphabetically within a severity band.

This keeps every recommendation auditable (still built from the curated
knowledge-base reason/action, per the explainability goal in
gap_recommendation.py) while making sure two gap instances essentially
never render identically unless their underlying context truly matches.
"""

import hashlib
import random

import numpy as np
import pandas as pd

from utils import achievement_rate, classify_severity

# ---------------------------------------------------------------------------
# 1. Context features from the raw time series
# ---------------------------------------------------------------------------

def compute_trend_features(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds, per (Department, Indicator) time series sorted by date:
      Achievement_Rate, Gap_Severity   (existing utils logic)
      Prev_Achievement_Rate, Change_Pts, Trend_Label
      Volatility_6mo                   (std of achievement rate, trailing 6 months)
      Consecutive_Gap_Months           (streak of non-"On Track" months ending at this row)
    """
    df = kpi_df.copy()
    df["Achievement_Rate"] = df.apply(
        lambda r: achievement_rate(r["Actual"], r["Target"], r["Direction"]), axis=1)
    df["Gap_Severity"] = df["Achievement_Rate"].apply(classify_severity)
    df = df.sort_values(["Department", "Indicator", "Year", "Month_Num"]).reset_index(drop=True)

    out_frames = []
    for (dept, indicator), g in df.groupby(["Department", "Indicator"], sort=False):
        g = g.sort_values(["Year", "Month_Num"]).copy()
        g["Prev_Achievement_Rate"] = g["Achievement_Rate"].shift(1)
        g["Change_Pts"] = (g["Achievement_Rate"] - g["Prev_Achievement_Rate"]).round(1)
        g["Volatility_6mo"] = g["Achievement_Rate"].rolling(6, min_periods=2).std().round(1)

        def trend_label(change):
            if pd.isna(change):
                return "New"
            if change > 2:
                return "Improving"
            if change < -2:
                return "Worsening"
            return "Stable"

        g["Trend_Label"] = g["Change_Pts"].apply(trend_label)

        streaks, streak = [], 0
        for sev in g["Gap_Severity"]:
            streak = streak + 1 if sev != "On Track" else 0
            streaks.append(streak)
        g["Consecutive_Gap_Months"] = streaks

        out_frames.append(g)

    return pd.concat(out_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. NLG phrase pools
# ---------------------------------------------------------------------------

_OPENERS = {
    "Critical Gap": [
        "Performance on {indicator} is critically off target at {rate}% of goal.",
        "{indicator} is in critical breach of target, currently achieving only {rate}%.",
        "A critical gap has opened on {indicator}, sitting at {rate}% of the target level.",
    ],
    "Moderate Gap": [
        "{indicator} is moderately underperforming at {rate}% of target.",
        "There is a moderate shortfall on {indicator}, currently at {rate}% of goal.",
        "{indicator} has slipped to {rate}% of target, a moderate gap worth addressing.",
    ],
    "Minor Gap": [
        "{indicator} is running slightly under target at {rate}% of goal.",
        "A minor gap is present on {indicator}, {rate}% of the target achieved.",
        "{indicator} is close to target but not quite there, at {rate}% achievement.",
    ],
    "On Track": [
        "{indicator} is on track at {rate}% of target.",
        "{indicator} is meeting or exceeding its target ({rate}%).",
    ],
}

_TREND_CLAUSES = {
    "Worsening": [
        "and the trend is worsening, down {change} points from last month.",
        "with performance actively deteriorating ({change} points versus last month).",
        "and this has gotten worse over the past month (change of {change} points).",
    ],
    "Improving": [
        "though it is improving, up {change} points from last month.",
        "but trending in the right direction, gaining {change} points month-over-month.",
        "with early signs of recovery, {change} points better than last month.",
    ],
    "Stable": [
        "and has been holding steady month-over-month.",
        "with performance essentially flat versus last month.",
    ],
    "New": [
        "based on this month's reading.",
    ],
}

_STREAK_CLAUSES = [
    "This is the {streak}th consecutive month below target — the underlying cause has not been resolved by prior action.",
    "Having persisted for {streak} consecutive months, this is no longer a one-off dip and needs a structural fix, not just monitoring.",
    "{streak} consecutive months in gap status suggests the root cause identified previously is still active.",
]

_VOLATILITY_CLAUSES = [
    "Performance on this indicator has also been unusually volatile recently (σ≈{vol} pts over 6 months), making the target harder to hold consistently.",
]

_ACTION_LEAD_IN = {
    "Critical Gap": ["URGENT — ", "IMMEDIATE ACTION — ", "Escalate now — "],
    "Moderate Gap": ["Recommended — ", "Prioritize this month — "],
    "Minor Gap": ["Monitor & improve — ", "Low-effort fix — "],
    "On Track": [""],
}

_ESCALATION_ACTION_SUFFIX = [
    " Given the {streak}-month streak, escalate to the facility PMT review for a root-cause deep-dive rather than repeating the standard fix.",
    " Because this has persisted {streak} months, assign a named owner and a 2-week follow-up checkpoint, not just the routine action list.",
]


def _seeded_rng(dept: str, indicator: str, year: int, month_num: int) -> random.Random:
    """Deterministic per-row RNG so the same gap always renders the same way, but
    different rows (different indicator/month/streak) pick different phrasing."""
    key = f"{dept}|{indicator}|{year}|{month_num}".encode()
    seed = int(hashlib.md5(key).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def urgency_score(severity: str, trend_label: str, streak: int, volatility: float) -> int:
    """0-100 composite used to re-prioritize the table beyond a flat severity band."""
    base = {"Critical Gap": 70, "Moderate Gap": 45, "Minor Gap": 20, "On Track": 0}[severity]
    trend_adj = {"Worsening": 15, "Stable": 5, "Improving": -10, "New": 0}[trend_label]
    streak_adj = min(streak * 3, 20)
    vol_adj = min((volatility or 0) / 2, 10)
    return int(np.clip(base + trend_adj + streak_adj + vol_adj, 0, 100))


# ---------------------------------------------------------------------------
# 3. Row-level composer
# ---------------------------------------------------------------------------

def generate_dynamic_recommendation(row: dict, kb_reason: str, kb_action: str) -> dict:
    """
    row must contain: Department, Indicator, Achievement_Rate, Gap_Severity,
    Change_Pts, Trend_Label, Consecutive_Gap_Months, Volatility_6mo.
    Returns dict(reason, action, urgency_score, trend, streak).
    """
    severity = row["Gap_Severity"]
    trend = row["Trend_Label"]
    streak = int(row["Consecutive_Gap_Months"])
    vol = row.get("Volatility_6mo")
    rng = _seeded_rng(row["Department"], row["Indicator"], row["Year"], row["Month_Num"])

    if severity == "On Track":
        opener = rng.choice(_OPENERS["On Track"]).format(
            indicator=row["Indicator"].replace("_", " "), rate=row["Achievement_Rate"])
        return dict(reason=opener + " Maintain current practices; continue monthly monitoring to sustain performance.",
                    action="Maintain current practices; continue monthly monitoring to sustain performance.",
                    urgency_score=0, trend=trend, streak=streak)

    opener = rng.choice(_OPENERS[severity]).format(
        indicator=row["Indicator"].replace("_", " "), rate=row["Achievement_Rate"])

    change_val = row.get("Change_Pts")
    change_str = f"{abs(change_val):.1f}" if pd.notna(change_val) else "0.0"
    trend_clause = rng.choice(_TREND_CLAUSES[trend]).format(change=change_str)

    sentence_parts = [opener, trend_clause, kb_reason]
    if streak >= 3:
        sentence_parts.append(rng.choice(_STREAK_CLAUSES).format(streak=streak))
    if vol is not None and pd.notna(vol) and vol >= 8 and rng.random() < 0.6:
        sentence_parts.append(rng.choice(_VOLATILITY_CLAUSES).format(vol=round(vol, 1)))

    reason_text = " ".join(sentence_parts)

    action_lead_in = rng.choice(_ACTION_LEAD_IN[severity])
    action_text = action_lead_in + kb_action
    if streak >= 4:
        action_text += rng.choice(_ESCALATION_ACTION_SUFFIX).format(streak=streak)

    score = urgency_score(severity, trend, streak, vol)

    return dict(reason=reason_text, action=action_text, urgency_score=score, trend=trend, streak=streak)
