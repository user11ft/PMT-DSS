"""
forecasting.py
--------------
Forecasts next-month performance for every (Department, Indicator)
series in the KPI dataset.

Design choice (why this approach, for the hackathon writeup):
  Each series only has ~36 monthly points, which rules out heavy
  deep-learning forecasters (would overfit badly). Instead we treat
  model *selection* itself as the innovation: for every indicator we
  fit three candidate models --
    1. Naive persistence   (next month = last month)
    2. Linear trend+season (time index + month-of-year Fourier terms)
    3. Random Forest       (lag features + rolling stats)
  -- evaluate each on a walk-forward holdout (last 6 months), and
  automatically keep whichever generalizes best for THAT indicator.
  This is far more robust for a small, heterogeneous multi-KPI
  dataset than forcing one model architecture on every series.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

HOLDOUT_MONTHS = 6


def _build_features(values: np.ndarray) -> pd.DataFrame:
    """Build a feature table (time trend, seasonal Fourier terms, lags, rolling stats)."""
    n = len(values)
    t = np.arange(n)
    month_of_year = (t % 12) + 1  # relies on series starting in January; fine for this dataset
    df = pd.DataFrame({
        "t": t,
        "month_sin": np.sin(2 * np.pi * month_of_year / 12),
        "month_cos": np.cos(2 * np.pi * month_of_year / 12),
        "y": values,
    })
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df["lag3"] = df["y"].shift(3)
    df["roll_mean_3"] = df["y"].shift(1).rolling(3).mean()
    df["roll_std_3"] = df["y"].shift(1).rolling(3).std()
    return df


FEATURE_COLS = ["t", "month_sin", "month_cos", "lag1", "lag2", "lag3", "roll_mean_3", "roll_std_3"]


def _evaluate_candidates(df: pd.DataFrame):
    """Walk-forward evaluate Naive / LinearRegression / RandomForest on the last HOLDOUT_MONTHS."""
    df = df.dropna().reset_index(drop=True)
    n = len(df)
    if n < HOLDOUT_MONTHS + 6:
        # too short for a proper holdout -> just use naive
        return "Naive", None, None

    train, test = df.iloc[: n - HOLDOUT_MONTHS], df.iloc[n - HOLDOUT_MONTHS:]
    results = {}

    # 1. Naive persistence
    naive_pred = train["y"].iloc[-1]
    results["Naive"] = mean_absolute_error(test["y"], [naive_pred] * len(test))

    # 2. Linear trend + seasonal
    lr = LinearRegression().fit(train[["t", "month_sin", "month_cos"]], train["y"])
    lr_pred = lr.predict(test[["t", "month_sin", "month_cos"]])
    results["LinearTrendSeasonal"] = mean_absolute_error(test["y"], lr_pred)

    # 3. Random Forest on lag + rolling features
    rf = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(
        train[FEATURE_COLS], train["y"])
    rf_pred = rf.predict(test[FEATURE_COLS])
    results["RandomForest"] = mean_absolute_error(test["y"], rf_pred)

    best_model = min(results, key=results.get)
    return best_model, results[best_model], results


def forecast_series(values: np.ndarray):
    """Fit the best model (chosen via holdout MAE) on the FULL series and predict month t+1."""
    df_full = _build_features(values)
    best_model, best_mae, all_mae = _evaluate_candidates(df_full)

    df_clean = df_full.dropna().reset_index(drop=True)
    last_row = df_full.iloc[[-1]]  # most recent point, used to build next-step features
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

    if best_model == "Naive" or df_clean.empty:
        prediction = values[-1]
        residual_std = np.std(values[-6:]) if len(values) >= 6 else np.std(values)
    elif best_model == "LinearTrendSeasonal":
        model = LinearRegression().fit(df_clean[["t", "month_sin", "month_cos"]], df_clean["y"])
        prediction = float(model.predict(next_features[["t", "month_sin", "month_cos"]])[0])
        residuals = df_clean["y"] - model.predict(df_clean[["t", "month_sin", "month_cos"]])
        residual_std = float(residuals.std())
    else:  # RandomForest
        model = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42).fit(
            df_clean[FEATURE_COLS], df_clean["y"])
        prediction = float(model.predict(next_features[FEATURE_COLS])[0])
        residuals = df_clean["y"] - model.predict(df_clean[FEATURE_COLS])
        residual_std = float(residuals.std())

    ci95 = 1.96 * residual_std
    return dict(
        model_used=best_model,
        holdout_mae=round(best_mae, 2) if best_mae is not None else None,
        prediction=round(float(prediction), 2),
        ci_low=round(float(prediction - ci95), 2),
        ci_high=round(float(prediction + ci95), 2),
    )


def forecast_all_indicators(kpi_df: pd.DataFrame) -> pd.DataFrame:
    """Run forecast_series() for every Department x Indicator group. Returns a summary table."""
    rows = []
    kpi_df = kpi_df.sort_values(["Department", "Indicator", "Year", "Month_Num"])
    for (dept, indicator), g in kpi_df.groupby(["Department", "Indicator"], sort=False):
        g = g.sort_values(["Year", "Month_Num"])
        values = g["Actual"].to_numpy()
        result = forecast_series(values)
        rows.append(dict(
            Department=dept,
            Indicator=indicator,
            Unit=g["Unit"].iloc[0],
            Direction=g["Direction"].iloc[0],
            Target=g["Target"].iloc[0],
            Last_Actual=values[-1],
            Predicted_Next_Month=result["prediction"],
            CI_Low=result["ci_low"],
            CI_High=result["ci_high"],
            Model_Used=result["model_used"],
            Holdout_MAE=result["holdout_mae"],
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    kpi_df = pd.read_csv("../data/ethiopia_health_kpis_monthly.csv") \
        if __package__ else pd.read_csv("data/ethiopia_health_kpis_monthly.csv")
    summary = forecast_all_indicators(kpi_df)
    pd.set_option("display.width", 160)
    print(summary.to_string(index=False))
