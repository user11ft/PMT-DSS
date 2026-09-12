"""
utils.py
--------
Shared, single-source-of-truth helpers so every module (data
generation, gap analysis, satisfaction analysis, the Streamlit app)
agrees on how "percent of target achieved" and "gap severity" are
defined.
"""

import pandas as pd

# Achievement-rate bands -> gap severity label.
# Tune these thresholds to match your facility's actual PMT policy.
SEVERITY_BANDS = [
    (100.0, float("inf"), "On Track"),
    (90.0, 100.0, "Minor Gap"),
    (75.0, 90.0, "Moderate Gap"),
    (float("-inf"), 75.0, "Critical Gap"),
]


def achievement_rate(actual: float, target: float, direction: str) -> float:
    """
    Percent of target achieved, adjusted for whether higher or lower
    is better. >=100 always means "target met or exceeded" regardless
    of direction, which keeps severity banding consistent across
    every indicator type (rates, counts, wait times, etc.).
    """
    if target == 0:
        return 100.0
    if direction == "higher":
        return round(actual / target * 100, 1)
    else:  # lower_is_better
        return round(target / actual * 100, 1) if actual > 0 else 100.0


def classify_severity(rate: float) -> str:
    for low, high, label in SEVERITY_BANDS:
        if low <= rate < high:
            return label
    return "Critical Gap"


def add_achievement_columns(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Adds Achievement_Rate and Gap_Severity columns to a raw KPI dataframe."""
    df = kpi_df.copy()
    df["Achievement_Rate"] = df.apply(
        lambda r: achievement_rate(r["Actual"], r["Target"], r["Direction"]), axis=1
    )
    df["Gap_Severity"] = df["Achievement_Rate"].apply(classify_severity)
    return df


def department_monthly_performance(kpi_df: pd.DataFrame, cap: float = 130.0) -> pd.DataFrame:
    """
    Average direction-adjusted achievement rate per Department x Month.
    This is the bridge that links the KPI dataset to the satisfaction
    dataset: "how was this department objectively performing that month?"
    `cap` limits how much a single over-performing indicator can pull the
    department average up (a facility hitting 300% of one small target
    shouldn't mask problems in three other indicators).
    """
    df = add_achievement_columns(kpi_df)
    df["Achievement_Rate_Capped"] = df["Achievement_Rate"].clip(upper=cap)
    perf = (df.groupby(["Department", "Year", "Month_Num"])["Achievement_Rate_Capped"]
              .mean().reset_index().rename(columns={"Achievement_Rate_Capped": "Perf_Score"}))
    return perf
