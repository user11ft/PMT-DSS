"""
app.py
------
AI-Based KPI Monitoring & PMT Decision Support System
Streamlit dashboard - deployable as-is to Streamlit Community Cloud.

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub -> share.streamlit.io -> New app
               -> repo, branch, main file path = app.py
"""

import os
import pickle
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from utils import department_monthly_performance  # noqa: E402
from forecasting import forecast_series  # noqa: E402
from gap_recommendation import build_gap_report  # noqa: E402
from satisfaction_analysis import (  # noqa: E402
    satisfaction_scores, domain_kpi_correlation, driver_analysis, generate_insight_text,
)
from train_and_export import predict_with_artifact  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_ARTIFACTS_PATH = os.path.join(BASE_DIR, "model_artifacts.pkl")


st.set_page_config(page_title="Health Sector Performance & Gap Analysis Dashboard",
                    page_icon="🩺", layout="wide")

# ---------------------------------------------------------------------------
# Styling (mirrors the hackathon mockup's navy header)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {background-color:#0b3d66; padding:18px 24px; border-radius:10px;
              color:white; margin-bottom:18px;}
.main-header h1 {margin:0; font-size:1.4rem;}
.severity-critical {background-color:#f8d7da; color:#842029; padding:2px 8px; border-radius:6px;}
.severity-moderate {background-color:#fff3cd; color:#664d03; padding:2px 8px; border-radius:6px;}
.severity-minor    {background-color:#e2f0d9; color:#3a5a1f; padding:2px 8px; border-radius:6px;}
.severity-ontrack  {background-color:#d1e7dd; color:#0f5132; padding:2px 8px; border-radius:6px;}
</style>
<div class="main-header"><h1>🩺 Health Sector Performance & Gap Analysis Dashboard</h1></div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
# The app first looks for the CSV files in:
#   1. data/
#   2. the repository root
# If a CSV is not found, the user can upload it from the Streamlit sidebar.
# This prevents FileNotFoundError on Streamlit Cloud.

def find_data_file(filename):
    possible_paths = [
        os.path.join(DATA_DIR, filename),
        os.path.join(BASE_DIR, filename),
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    return None


st.sidebar.markdown("### Data Files")

uploaded_kpi = st.sidebar.file_uploader(
    "Upload KPI CSV",
    type=["csv"],
    help="Upload ethiopia_health_kpis_monthly.csv if it is not available in GitHub."
)

uploaded_survey = st.sidebar.file_uploader(
    "Upload Client Satisfaction CSV",
    type=["csv"],
    help="Upload client_satisfaction_survey.csv if it is not available in GitHub."
)


@st.cache_data
def load_data(kpi_bytes=None, survey_bytes=None):

    kpi_filename = "ethiopia_health_kpis_monthly.csv"
    survey_filename = "client_satisfaction_survey.csv"

    # Use uploaded KPI data when provided.
    if kpi_bytes is not None:
        kpi_df = pd.read_csv(kpi_bytes)
    else:
        kpi_path = find_data_file(kpi_filename)

        if kpi_path is None:
            st.error(
                f"❌ KPI dataset not found: {kpi_filename}"
            )
            st.info(
                "Please upload the KPI CSV using the sidebar, "
                "or add the file to the data/ folder in GitHub."
            )
            st.stop()

        kpi_df = pd.read_csv(kpi_path)

    # Use uploaded survey data when provided.
    if survey_bytes is not None:
        survey_df = pd.read_csv(survey_bytes)
    else:
        survey_path = find_data_file(survey_filename)

        if survey_path is None:
            st.error(
                f"❌ Survey dataset not found: {survey_filename}"
            )
            st.info(
                "Please upload the survey CSV using the sidebar, "
                "or add the file to the data/ folder in GitHub."
            )
            st.stop()

        survey_df = pd.read_csv(survey_path)

    return kpi_df, survey_df


# Convert uploaded files to bytes before passing them to the cached function.
# This makes the Streamlit cache stable and avoids problems with UploadedFile objects.
kpi_bytes = uploaded_kpi.getvalue() if uploaded_kpi is not None else None
survey_bytes = uploaded_survey.getvalue() if uploaded_survey is not None else None

@st.cache_resource
def load_model_artifacts():
    """Load the pre-trained model bundle (run `python src/train_and_export.py` to build/refresh it)."""
    if os.path.exists(MODEL_ARTIFACTS_PATH):
        with open(MODEL_ARTIFACTS_PATH, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def run_forecasts(kpi_df, _bundle):
    if _bundle is not None:
        # Fast path: reuse the fitted estimators saved in model_artifacts.pkl instead of retraining.
        rows = []
        for (dept, indicator), g in kpi_df.groupby(["Department", "Indicator"], sort=False):
            g = g.sort_values(["Year", "Month_Num"])
            values = g["Actual"].to_numpy()
            artifact = _bundle["forecast_models"].get((dept, indicator))
            if artifact is None:
                result = forecast_series(values)  # new indicator not in the pickled bundle -> fit live
            else:
                result = predict_with_artifact(artifact, values)
            rows.append(dict(Department=dept, Indicator=indicator, Unit=g["Unit"].iloc[0],
                              Direction=g["Direction"].iloc[0], Target=g["Target"].iloc[0],
                              Last_Actual=values[-1], **result))
        return pd.DataFrame(rows)

    # Fallback: no pkl found, train everything live (slower cold start).
    rows = []
    for (dept, indicator), g in kpi_df.groupby(["Department", "Indicator"], sort=False):
        g = g.sort_values(["Year", "Month_Num"])
        values = g["Actual"].to_numpy()
        result = forecast_series(values)
        rows.append(dict(Department=dept, Indicator=indicator, Unit=g["Unit"].iloc[0],
                          Direction=g["Direction"].iloc[0], Target=g["Target"].iloc[0],
                          Last_Actual=values[-1], **result))
    return pd.DataFrame(rows)


@st.cache_data
def run_gap_report(kpi_df):
    # Rule-based + TF-IDF fallback; cheap enough to always compute live on the current month's data.
    return build_gap_report(kpi_df)


@st.cache_data
def run_driver_analysis(survey_df, kpi_df, _bundle):
    if _bundle is not None:
        result = dict(model=_bundle["driver_model"], test_accuracy=_bundle["driver_test_accuracy"],
                       driver_ranking=_bundle["driver_ranking"], domain_importance=_bundle["domain_importance"])
        return result, _bundle["corr_table"], _bundle["insight_text"]

    result = driver_analysis(survey_df, kpi_df)
    corr = domain_kpi_correlation(survey_df, kpi_df)
    insight = generate_insight_text(result, corr)
    return result, corr, insight


kpi_df, survey_df = load_data(kpi_bytes, survey_bytes)
model_bundle = load_model_artifacts()
forecast_df = run_forecasts(kpi_df, model_bundle)
gap_report = run_gap_report(kpi_df)
driver_result, corr_table, insight_text = run_driver_analysis(survey_df, kpi_df, model_bundle)

if model_bundle is None:
    st.sidebar.warning("model_artifacts.pkl not found — training models live this run "
                        "(slower). Run `python src/train_and_export.py` and redeploy to speed this up.")
else:
    st.sidebar.caption(f"✅ Loaded pre-trained models (trained {model_bundle['trained_at'][:19].replace('T', ' ')} UTC)")

DEPARTMENTS = ["Overview", "Emergency", "Pharmacy", "Laboratory", "Delivery Ward", "Outpatient"]
DEPT_ICONS = {"Overview": "🏥", "Emergency": "🚑", "Pharmacy": "💊", "Laboratory": "🧪",
              "Delivery Ward": "🛏️", "Outpatient": "🚻"}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Departments")
view = st.sidebar.radio(
    "Select a view", DEPARTMENTS,
    format_func=lambda d: f"{DEPT_ICONS[d]}  {d}", label_visibility="collapsed")

st.sidebar.markdown("### AI Insights & Predictions")
ai_view = st.sidebar.radio(
    "AI tools", ["(none)", "📈 Predictive Report", "🧭 Resource Allocation"],
    label_visibility="collapsed")

SEVERITY_CLASS = {"Critical Gap": "severity-critical", "Moderate Gap": "severity-moderate",
                   "Minor Gap": "severity-minor", "On Track": "severity-ontrack"}


TREND_ICON = {"Improving": "📈 Improving", "Worsening": "📉 Worsening", "Stable": "➡️ Stable", "New": "🆕 New"}


def render_gap_table(df: pd.DataFrame):
    cols = ["Department", "Indicator", "Gap_Severity"]
    if "Trend" in df.columns:
        cols += ["Trend", "Consecutive_Months_In_Gap", "Urgency_Score"]
    cols += ["Achievement_Rate", "Reason_For_Gap", "Action_Plan_Recommendation"]
    show = df[cols].copy()
    show["Gap_Severity"] = show.apply(
        lambda r: f'<span class="{SEVERITY_CLASS[r.Gap_Severity]}">{r.Gap_Severity}</span>', axis=1)
    if "Trend" in show.columns:
        show["Trend"] = show["Trend"].map(TREND_ICON).fillna(show["Trend"])
        show = show.rename(columns={"Consecutive_Months_In_Gap": "Months In Gap",
                                     "Urgency_Score": "Urgency (0-100, AI-prioritized)"})
    st.write(show.to_html(escape=False, index=False), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AI Insights sidebar pages (take priority over department view)
# ---------------------------------------------------------------------------
if ai_view == "📈 Predictive Report":
    st.subheader("📈 Predictive Report — next-month forecast per indicator")
    st.caption("Each indicator's forecast model (Naive / Linear trend+seasonal / Random Forest) "
               "is auto-selected per series using a 6-month holdout — see src/forecasting.py.")
    st.dataframe(forecast_df[["Department", "Indicator", "Unit", "Last_Actual",
                               "prediction", "ci_low", "ci_high", "model_used", "holdout_mae"]]
                 .rename(columns={"prediction": "Predicted_Next_Month",
                                  "ci_low": "CI_Low", "ci_high": "CI_High",
                                  "model_used": "Model_Used", "holdout_mae": "Holdout_MAE"}),
                 width='stretch', hide_index=True)

    st.subheader("🔎 Satisfaction driver analysis")
    st.info(insight_text)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Question-domain importance (ML model)**")
        st.bar_chart(driver_result["domain_importance"])
    with c2:
        st.markdown("**Correlation with KPI performance**")
        st.dataframe(corr_table, hide_index=True, width='stretch')

elif ai_view == "🧭 Resource Allocation":
    st.subheader("🧭 Resource Allocation Priority")
    st.caption("Departments ranked by count of Critical/Moderate gaps this month — "
               "a data-driven starting point for where to direct staff, budget, or supplies first.")
    counts = (gap_report.groupby(["Department", "Gap_Severity"]).size()
              .unstack(fill_value=0)
              .reindex(columns=["Critical Gap", "Moderate Gap", "Minor Gap", "On Track"], fill_value=0))
    if "Urgency_Score" in gap_report.columns:
        counts["Priority_Score"] = gap_report.groupby("Department")["Urgency_Score"].sum()
        st.caption("Priority score now sums each department's AI Urgency_Score "
                   "(severity + worsening trend + persistence), not just a flat gap count.")
    else:
        counts["Priority_Score"] = counts["Critical Gap"] * 3 + counts["Moderate Gap"] * 1
    counts = counts.sort_values("Priority_Score", ascending=False)
    st.bar_chart(counts[["Critical Gap", "Moderate Gap", "Minor Gap", "On Track"]])
    st.dataframe(counts, width='stretch')
    st.markdown("**Suggested priority order:** " + " → ".join(counts.index.tolist()))

else:
    # -----------------------------------------------------------------
    # Department views (Overview or one specific department)
    # -----------------------------------------------------------------
    dept_filter = None if view == "Overview" else view
    dept_gap = gap_report if dept_filter is None else gap_report[gap_report["Department"] == dept_filter]
    dept_forecast = forecast_df if dept_filter is None else forecast_df[forecast_df["Department"] == dept_filter]

    # --- top KPI cards ---
    col1, col2, col3 = st.columns(3)

    if view == "Overview":
        opd = kpi_df[kpi_df["Indicator"] == "Total_OPD_Visits"].sort_values(["Year", "Month_Num"])
        last_visits = opd["Actual"].iloc[-1]
        prev_visits = opd["Actual"].iloc[-2]
        pct_change = (last_visits - prev_visits) / prev_visits * 100
        pred_row = forecast_df[forecast_df["Indicator"] == "Total_OPD_Visits"].iloc[0]

        col1.metric("Total Patient Visits (This Month)", f"{last_visits:,.0f}", f"{pct_change:+.1f}% vs last month")
        col2.metric("Predicted Visits (Next Month)", f"{pred_row['prediction']:,.0f}",
                    help=f"Model: {pred_row['model_used']} | 95% CI: {pred_row['ci_low']:,.0f}-{pred_row['ci_high']:,.0f}")
        overall_sat = satisfaction_scores(survey_df, ["Department"])
        avg_5pt = (survey_df["Satisfaction_Response"]
                   .map({"Satisfied": 5, "Neutral": 3, "Unsatisfied": 1}).mean())
        stars = "⭐" * round(avg_5pt) + "☆" * (5 - round(avg_5pt))
        col3.metric("Avg. Satisfaction Rate", f"{avg_5pt:.1f} / 5", stars)
    else:
        dept_indicators = kpi_df[kpi_df["Department"] == dept_filter]["Indicator"].unique()[:3]
        for col, indicator in zip([col1, col2, col3], dept_indicators):
            row = dept_forecast[dept_forecast["Indicator"] == indicator]
            g = kpi_df[(kpi_df["Department"] == dept_filter) & (kpi_df["Indicator"] == indicator)] \
                .sort_values(["Year", "Month_Num"])
            last_val = g["Actual"].iloc[-1]
            unit = g["Unit"].iloc[0]
            label = indicator.replace("_", " ")
            if not row.empty:
                col.metric(label, f"{last_val:,.1f} {unit}",
                           f"→ {row['prediction'].iloc[0]:,.1f} predicted next month")
            else:
                col.metric(label, f"{last_val:,.1f} {unit}")

    st.markdown("---")

    # --- gap analysis table ---
    st.subheader("Gap Analysis & Recommendations" + ("" if dept_filter is None else f" — {dept_filter}"))
    render_gap_table(dept_gap)

    st.markdown("---")

    # --- trend chart ---
    st.subheader("Actual vs. Predicted Trend")
    trend_indicator = "Total_OPD_Visits" if view == "Overview" else kpi_df[
        kpi_df["Department"] == dept_filter]["Indicator"].iloc[0]
    series = kpi_df[kpi_df["Indicator"] == trend_indicator].sort_values(["Year", "Month_Num"])
    labels = [f"{r.Year}-{r.Month_Num:02d}" for r in series.itertuples()]
    pred_row = forecast_df[forecast_df["Indicator"] == trend_indicator].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=series["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=labels, y=[series["Target"].iloc[0]] * len(labels),
                              mode="lines", name="Target", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=labels + [f"Next month"], y=list(series["Actual"]) + [pred_row["prediction"]],
                              mode="lines+markers", name="Predicted", line=dict(dash="dash", color="orange")))
    fig.update_layout(height=380, title=trend_indicator.replace("_", " "),
                       margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    # --- satisfaction breakdown ---
    st.subheader("Client Satisfaction Breakdown")
    sat_group = ["Department"] if dept_filter is None else ["Question_Domain"]
    sat_data = survey_df if dept_filter is None else survey_df[survey_df["Department"] == dept_filter]
    scores = satisfaction_scores(sat_data, sat_group)
    st.dataframe(scores, width='stretch', hide_index=True)

    fig2 = go.Figure()
    for col, color in [("Pct_Satisfied", "#2e7d32"), ("Pct_Neutral", "#f9a825"), ("Pct_Unsatisfied", "#c62828")]:
        fig2.add_trace(go.Bar(x=scores[sat_group[0]], y=scores[col], name=col.replace("Pct_", ""), marker_color=color))
    fig2.update_layout(barmode="stack", height=340, margin=dict(l=10, r=10, t=20, b=10),
                        legend=dict(orientation="h"))
    st.plotly_chart(fig2, width='stretch')
