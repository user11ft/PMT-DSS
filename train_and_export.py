"""
train_and_export.py
--------------------
Runs the full KPI-PMT pipeline ONCE (forecasting model selection per
indicator, satisfaction driver classifier, gap report, correlations)
and pickles everything into a single artifact file:

    model_artifacts.pkl

app.py then loads this file directly instead of retraining every time
Streamlit boots -> much faster cold-start on Streamlit Community Cloud,
and a genuine ".pkl model file" to point to for the "AI model" deliverable.

Re-run this script any time the underlying CSVs change:

    python src/train_and_export.py
"""

import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(__file__))
from forecasting import _build_features, _evaluate_candidates, FEATURE_COLS  # noqa: E402
from gap_recommendation import build_gap_report  # noqa: E402
from satisfaction_analysis import (  # noqa: E402
    satisfaction_scores, domain_kpi_correlation, driver_analysis, generate_insight_text,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")


def fit_forecast_model_for_series(values: np.ndarray) -> dict:
    """
    Same model-selection logic as forecasting.forecast_series(), but
    returns the FITTED estimator object (plus everything needed to
    build the next-step feature row later) so it can be pickled and
    reused at inference time without retraining.
    """
    df_full = _build_features(values)
    best_model, best_mae, all_mae = _evaluate_candidates(df_full)
    df_clean = df_full.dropna().reset_index(drop=True)

    artifact = dict(model_type=best_model, holdout_mae=best_mae, all_holdout_mae=all_mae,
                     estimator=None, n_points=len(values))

    if best_model == "Naive" or df_clean.empty:
        artifact["residual_std"] = float(np.std(values[-6:]) if len(values) >= 6 else np.std(values))
    elif best_model == "LinearTrendSeasonal":
        model = LinearRegression().fit(df_clean[["t", "month_sin", "month_cos"]], df_clean["y"])
        residuals = df_clean["y"] - model.predict(df_clean[["t", "month_sin", "month_cos"]])
        artifact["estimator"] = model
        artifact["residual_std"] = float(residuals.std())
    else:  # RandomForest
        model = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(
            df_clean[FEATURE_COLS], df_clean["y"])
        residuals = df_clean["y"] - model.predict(df_clean[FEATURE_COLS])
        artifact["estimator"] = model
        artifact["residual_std"] = float(residuals.std())

    return artifact


def predict_with_artifact(artifact: dict, values: np.ndarray) -> dict:
    """Use a fitted artifact (from fit_forecast_model_for_series) to predict month t+1."""
    next_t = len(values)
    next_month = (next_t % 12) + 1
    next_features = pd.DataFrame({
        "t": [next_t],
        "month_sin": [np.sin(2 * np.pi * next_month / 12)],
        "month_cos": [np.cos(2 * np.pi * next_month / 12)],
        "lag1": [values[-1]],
        "lag2": [values[-2] if len(values) > 1 else values[-1]],
        "lag3": [values[-3] if len(values) > 2 else values[-1]],
        "roll_mean_3": [np.mean(values[-3:])],
        "roll_std_3": [np.std(values[-3:])],
    })

    if artifact["model_type"] == "Naive" or artifact["estimator"] is None:
        prediction = float(values[-1])
    elif artifact["model_type"] == "LinearTrendSeasonal":
        prediction = float(artifact["estimator"].predict(next_features[["t", "month_sin", "month_cos"]])[0])
    else:
        prediction = float(artifact["estimator"].predict(next_features[FEATURE_COLS])[0])

    ci95 = 1.96 * artifact["residual_std"]
    return dict(
        model_used=artifact["model_type"],
        holdout_mae=round(artifact["holdout_mae"], 2) if artifact["holdout_mae"] is not None else None,
        prediction=round(prediction, 2),
        ci_low=round(prediction - ci95, 2),
        ci_high=round(prediction + ci95, 2),
    )


def main():
    print("Loading data...")
    kpi_df = pd.read_csv(os.path.join(DATA_DIR, "ethiopia_health_kpis_monthly.csv"))
    survey_df = pd.read_csv(os.path.join(DATA_DIR, "client_satisfaction_survey.csv"))

    # ---------------- Forecasting: fit + pickle one model per (Department, Indicator) ----------------
    print("Fitting forecast models per indicator...")
    forecast_models = {}
    forecast_rows = []
    kpi_sorted = kpi_df.sort_values(["Department", "Indicator", "Year", "Month_Num"])
    for (dept, indicator), g in kpi_sorted.groupby(["Department", "Indicator"], sort=False):
        g = g.sort_values(["Year", "Month_Num"])
        values = g["Actual"].to_numpy()
        artifact = fit_forecast_model_for_series(values)
        forecast_models[(dept, indicator)] = artifact

        result = predict_with_artifact(artifact, values)
        forecast_rows.append(dict(
            Department=dept, Indicator=indicator, Unit=g["Unit"].iloc[0],
            Direction=g["Direction"].iloc[0], Target=g["Target"].iloc[0],
            Last_Actual=values[-1], **result,
        ))
    forecast_df = pd.DataFrame(forecast_rows)
    print(f"  fitted {len(forecast_models)} indicator models "
          f"({(forecast_df.model_used == 'RandomForest').sum()} RF, "
          f"{(forecast_df.model_used == 'LinearTrendSeasonal').sum()} Linear, "
          f"{(forecast_df.model_used == 'Naive').sum()} Naive)")

    # ---------------- Gap analysis (rule-based + TF-IDF fallback; deterministic, still bundled) -------
    print("Building latest gap report...")
    gap_report = build_gap_report(kpi_df)

    # ---------------- Satisfaction: driver classifier + correlation table -----------------------------
    print("Training satisfaction driver classifier...")
    driver_result = driver_analysis(survey_df, kpi_df)
    corr_table = domain_kpi_correlation(survey_df, kpi_df)
    insight_text = generate_insight_text(driver_result, corr_table)
    sat_by_department = satisfaction_scores(survey_df, ["Department"])

    bundle = dict(
        version=1,
        trained_at=datetime.now(timezone.utc).isoformat(),
        sklearn_version=__import__("sklearn").__version__,
        pandas_version=pd.__version__,

        # forecasting
        forecast_models=forecast_models,        # {(dept, indicator): fitted artifact dict}
        forecast_df=forecast_df,                # ready-to-display summary table

        # prescriptive
        gap_report=gap_report,

        # satisfaction / diagnostic
        driver_model=driver_result["model"],           # fitted RandomForestClassifier
        driver_feature_columns=list(driver_result["model"].feature_names_in_)
            if hasattr(driver_result["model"], "feature_names_in_") else None,
        driver_test_accuracy=driver_result["test_accuracy"],
        driver_ranking=driver_result["driver_ranking"],
        domain_importance=driver_result["domain_importance"],
        corr_table=corr_table,
        insight_text=insight_text,
        sat_by_department=sat_by_department,
    )

    with open(OUT_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"\nSaved {OUT_PATH} ({size_kb:.1f} KB)")
    print("Load it in app.py / anywhere with:")
    print("  import pickle; bundle = pickle.load(open('model_artifacts.pkl', 'rb'))")


if __name__ == "__main__":
    main()
