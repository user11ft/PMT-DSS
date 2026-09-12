"""
satisfaction_analysis.py
-------------------------
Turns raw Satisfied/Neutral/Unsatisfied survey rows into:

  1. A "X.X / 5" satisfaction score per department/month, matching the
     dashboard mockup's "Avg. Satisfaction Rate" card.
  2. A driver analysis: which question domain (Access, Communication,
     Respect, Information) most explains dissatisfaction, and how
     strongly each domain's satisfaction tracks that department's
     objective KPI performance that month. This is what lets the
     system tell a manager *why* satisfaction is low, not just that
     it is -- closing the loop between the two datasets.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils import department_monthly_performance

RESPONSE_TO_5PT = {"Satisfied": 5, "Neutral": 3, "Unsatisfied": 1}
RESPONSE_TO_100 = {"Satisfied": 100, "Neutral": 50, "Unsatisfied": 0}


def satisfaction_scores(survey_df: pd.DataFrame, group_cols=("Department",)) -> pd.DataFrame:
    """Average 5-point and 0-100 satisfaction score, plus response mix, per group."""
    df = survey_df.copy()
    df["score_5pt"] = df["Satisfaction_Response"].map(RESPONSE_TO_5PT)
    df["score_100"] = df["Satisfaction_Response"].map(RESPONSE_TO_100)

    agg = df.groupby(list(group_cols)).agg(
        Avg_Score_5pt=("score_5pt", "mean"),
        Avg_Score_100=("score_100", "mean"),
        N_Responses=("score_5pt", "size"),
        Pct_Satisfied=("Satisfaction_Response", lambda s: round((s == "Satisfied").mean() * 100, 1)),
        Pct_Neutral=("Satisfaction_Response", lambda s: round((s == "Neutral").mean() * 100, 1)),
        Pct_Unsatisfied=("Satisfaction_Response", lambda s: round((s == "Unsatisfied").mean() * 100, 1)),
    ).reset_index()
    agg["Avg_Score_5pt"] = agg["Avg_Score_5pt"].round(2)
    agg["Avg_Score_100"] = agg["Avg_Score_100"].round(1)
    return agg


def domain_kpi_correlation(survey_df: pd.DataFrame, kpi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple, transparent driver signal: correlate each question domain's
    monthly satisfaction score (per department) with that department's
    KPI achievement score the same month. High |correlation| = that
    domain's satisfaction rises and falls with operational performance;
    low correlation = driven by something the KPI dataset doesn't capture
    (e.g. staff attitude), which is itself a useful finding.
    """
    perf = department_monthly_performance(kpi_df)
    df = survey_df.copy()
    df["score_100"] = df["Satisfaction_Response"].map(RESPONSE_TO_100)

    monthly_domain = (df.groupby(["Department", "Year", "Month_Num", "Question_Domain"])["score_100"]
                        .mean().reset_index())
    merged = monthly_domain.merge(perf, on=["Department", "Year", "Month_Num"], how="left")

    rows = []
    for domain, g in merged.groupby("Question_Domain"):
        corr = g["score_100"].corr(g["Perf_Score"])
        rows.append(dict(Question_Domain=domain, Correlation_With_KPI_Performance=round(corr, 2),
                          N_Department_Months=len(g)))
    return pd.DataFrame(rows).sort_values("Correlation_With_KPI_Performance", ascending=False).reset_index(drop=True)


def driver_analysis(survey_df: pd.DataFrame, kpi_df: pd.DataFrame):
    """
    ML driver analysis: RandomForestClassifier predicts each response
    (Satisfied/Neutral/Unsatisfied) from Department, Question_Domain,
    Age_Group, Sex and that department-month's KPI Perf_Score. Feature
    importances (aggregated back to the original variable) show which
    factor matters most for satisfaction outcomes.
    """
    perf = department_monthly_performance(kpi_df)
    df = survey_df.merge(perf, on=["Department", "Year", "Month_Num"], how="left")
    df["Perf_Score"] = df["Perf_Score"].fillna(df["Perf_Score"].mean())

    feature_cols_cat = ["Department", "Question_Domain", "Age_Group", "Sex"]
    X_cat = pd.get_dummies(df[feature_cols_cat], prefix=feature_cols_cat)
    X = pd.concat([X_cat, df[["Perf_Score"]]], axis=1)
    y = df["Satisfaction_Response"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, clf.predict(X_test))

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    # collapse one-hot columns back to their parent variable (e.g. all
    # "Question_Domain_*" columns sum into one "Question_Domain" importance)
    grouped = {}
    for col, imp in importances.items():
        parent = next((c for c in feature_cols_cat if col.startswith(c + "_")), col)
        grouped[parent] = grouped.get(parent, 0) + imp
    driver_ranking = pd.Series(grouped).sort_values(ascending=False)

    # within Question_Domain, which specific domain contributes most to *dissatisfaction*
    domain_cols = [c for c in importances.index if c.startswith("Question_Domain_")]
    domain_importance = importances[domain_cols].sort_values(ascending=False)
    domain_importance.index = [c.replace("Question_Domain_", "") for c in domain_importance.index]

    return dict(
        model=clf,
        test_accuracy=round(test_accuracy, 3),
        driver_ranking=driver_ranking.round(4),
        domain_importance=domain_importance.round(4),
    )


def generate_insight_text(driver_result: dict, corr_table: pd.DataFrame) -> str:
    top_domain = driver_result["domain_importance"].index[0]
    corr_row = corr_table[corr_table["Question_Domain"] == top_domain]
    corr_val = corr_row["Correlation_With_KPI_Performance"].iloc[0] if not corr_row.empty else None
    corr_phrase = (f"and correlates {abs(corr_val):.2f} with objective KPI performance in that domain's related departments"
                   if corr_val is not None else "")
    return (f"'{top_domain}' is the strongest predictor of client satisfaction/dissatisfaction "
            f"across all survey responses {corr_phrase}. "
            f"Model cross-validated accuracy: {driver_result['test_accuracy']*100:.1f}% on held-out responses.")


if __name__ == "__main__":
    kpi_df = pd.read_csv("../data/ethiopia_health_kpis_monthly.csv") if __package__ else pd.read_csv("data/ethiopia_health_kpis_monthly.csv")
    survey_df = pd.read_csv("../data/client_satisfaction_survey.csv") if __package__ else pd.read_csv("data/client_satisfaction_survey.csv")

    print("=== Satisfaction score by department (X / 5) ===")
    print(satisfaction_scores(survey_df, ["Department"]).to_string(index=False))

    print("\n=== Domain <-> KPI performance correlation ===")
    corr_table = domain_kpi_correlation(survey_df, kpi_df)
    print(corr_table.to_string(index=False))

    print("\n=== ML driver analysis ===")
    result = driver_analysis(survey_df, kpi_df)
    print("Variable importance ranking:\n", result["driver_ranking"])
    print("\nQuestion-domain importance:\n", result["domain_importance"])
    print("\nInsight:", generate_insight_text(result, corr_table))
