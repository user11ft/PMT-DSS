"""
generate_data.py
-----------------
Generates two synthetic-but-realistic datasets for the
"AI-Based KPI Monitoring & PMT Decision Support System":

  1. data/ethiopia_health_kpis_monthly.csv
     Monthly Target vs Actual performance for 18 HMIS-style indicators
     across 6 department/service areas of a typical Ethiopian health
     center, over 36 months (Jan 2023 - Dec 2025).

  2. data/client_satisfaction_survey.csv
     Client-satisfaction survey responses (Satisfied / Neutral /
     Unsatisfied) across 8 questions in 4 domains (Access,
     Communication, Respect, Information), for the same 36 months.
     Response probabilities are DELIBERATELY linked to that
     department's KPI achievement rate that month, so the two
     datasets tell one connected story (this is what the driver
     -analysis model in src/satisfaction_analysis.py recovers).

NOTE: This is synthetic data built to be realistic for a hackathon
demo. Swap in real values by exporting from DHIS2 (KPI dataset) and
KoboToolbox (survey dataset) using the same column names.
"""

import numpy as np
import pandas as pd
from datetime import date
from utils import achievement_rate as _achievement_rate
from utils import department_monthly_performance

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

START_YEAR = 2023
N_MONTHS = 36  # Jan 2023 -> Dec 2025
MONTHS = pd.date_range(start=f"{START_YEAR}-01-01", periods=N_MONTHS, freq="MS")

# ---------------------------------------------------------------
# 1. KPI INDICATOR DEFINITIONS
# ---------------------------------------------------------------
# Each indicator has:
#   start / end   -> linear trend anchor values across the 36 months
#   noise_std     -> month-to-month random noise (std dev)
#   seasonal_amp  -> amplitude of an annual sine wave (0 = none)
#   seasonal_peak -> month (1-12) where the seasonal wave peaks
#   target        -> constant monthly target
#   direction     -> "higher" (bigger=better) or "lower" (smaller=better)
#   unit, category-> for labeling/grouping in the dashboard

INDICATORS = [
    # -- Emergency --------------------------------------------------
    dict(dept="Emergency", name="Avg_Triage_to_Treatment_Time_Min",
         category="Service Efficiency", unit="minutes", direction="lower",
         target=15, start=17, end=31, noise_std=1.8, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Emergency", name="Emergency_Case_Fatality_Rate_Pct",
         category="Quality & Safety", unit="%", direction="lower",
         target=1.0, start=1.1, end=1.3, noise_std=0.15, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Emergency", name="Emergency_Bed_Availability_Rate_Pct",
         category="Resource Availability", unit="%", direction="higher",
         target=90, start=82, end=54, noise_std=3.5, seasonal_amp=4, seasonal_peak=10),

    # -- Pharmacy -----------------------------------------------------
    dict(dept="Pharmacy", name="Essential_Medicine_Availability_Rate_Pct",
         category="Resource Availability", unit="%", direction="higher",
         target=95, start=91, end=69, noise_std=3.0, seasonal_amp=3, seasonal_peak=7),
    dict(dept="Pharmacy", name="Stockout_Rate_Pct",
         category="Resource Availability", unit="%", direction="lower",
         target=5, start=7, end=24, noise_std=2.2, seasonal_amp=3, seasonal_peak=7),
    dict(dept="Pharmacy", name="Avg_Prescription_Fill_Time_Min",
         category="Service Efficiency", unit="minutes", direction="lower",
         target=10, start=11, end=15, noise_std=1.2, seasonal_amp=0, seasonal_peak=1),

    # -- Laboratory ---------------------------------------------------
    dict(dept="Laboratory", name="Avg_Test_Turnaround_Time_Hours",
         category="Service Efficiency", unit="hours", direction="lower",
         target=4, start=5.0, end=8.8, noise_std=0.6, seasonal_amp=0.6, seasonal_peak=10),
    dict(dept="Laboratory", name="Lab_QC_Pass_Rate_Pct",
         category="Quality & Safety", unit="%", direction="higher",
         target=98, start=96.5, end=95.0, noise_std=1.0, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Laboratory", name="Sample_Rejection_Rate_Pct",
         category="Quality & Safety", unit="%", direction="lower",
         target=2, start=2.3, end=4.2, noise_std=0.5, seasonal_amp=0, seasonal_peak=1),

    # -- Delivery Ward --------------------------------------------------
    dict(dept="Delivery Ward", name="Institutional_Delivery_Rate_Pct",
         category="Maternal & Child Health", unit="%", direction="higher",
         target=90, start=78, end=88, noise_std=2.0, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Delivery Ward", name="Skilled_Birth_Attendance_Rate_Pct",
         category="Maternal & Child Health", unit="%", direction="higher",
         target=95, start=85, end=92, noise_std=1.8, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Delivery Ward", name="Delivery_Bed_Availability_Rate_Pct",
         category="Resource Availability", unit="%", direction="higher",
         target=90, start=81, end=58, noise_std=3.2, seasonal_amp=3, seasonal_peak=10),

    # -- Outpatient -----------------------------------------------------
    dict(dept="Outpatient", name="Total_OPD_Visits",
         category="Utilization", unit="count", direction="higher",
         target=13500, start=11200, end=15600, noise_std=450, seasonal_amp=900, seasonal_peak=10),
    dict(dept="Outpatient", name="ANC4_Coverage_Rate_Pct",
         category="Maternal & Child Health", unit="%", direction="higher",
         target=90, start=69, end=82, noise_std=2.2, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Outpatient", name="Full_Immunization_Coverage_Pct",
         category="Maternal & Child Health", unit="%", direction="higher",
         target=95, start=79, end=90, noise_std=2.0, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Outpatient", name="Avg_Outpatient_Waiting_Time_Min",
         category="Service Efficiency", unit="minutes", direction="lower",
         target=20, start=24, end=36, noise_std=2.5, seasonal_amp=3, seasonal_peak=10),

    # -- Facility-wide (HMIS data quality, ties to real DQA work) -------
    dict(dept="Facility-Wide", name="HMIS_Reporting_Completeness_Pct",
         category="Data Quality", unit="%", direction="higher",
         target=100, start=96, end=98, noise_std=1.5, seasonal_amp=0, seasonal_peak=1),
    dict(dept="Facility-Wide", name="HMIS_Reporting_Timeliness_Pct",
         category="Data Quality", unit="%", direction="higher",
         target=100, start=93, end=97, noise_std=2.0, seasonal_amp=0, seasonal_peak=1),
]


def simulate_series(ind):
    """Build one 36-point monthly series: linear trend + seasonal wave + noise."""
    t = np.arange(N_MONTHS)
    trend = np.linspace(ind["start"], ind["end"], N_MONTHS)
    if ind["seasonal_amp"] > 0:
        month_nums = MONTHS.month
        phase = 2 * np.pi * (month_nums - ind["seasonal_peak"]) / 12
        seasonal = ind["seasonal_amp"] * np.cos(phase)
    else:
        seasonal = np.zeros(N_MONTHS)
    noise = rng.normal(0, ind["noise_std"], N_MONTHS)
    values = trend + seasonal + noise

    # keep percentages within sane bounds, counts non-negative
    if ind["unit"] == "%":
        values = np.clip(values, 0, 100)
    else:
        values = np.clip(values, 0, None)
    return values


def build_kpi_dataset():
    rows = []
    for ind in INDICATORS:
        values = simulate_series(ind)
        for month_ts, actual in zip(MONTHS, values):
            rows.append(dict(
                Year=month_ts.year,
                Month_Num=month_ts.month,
                Month_Name=month_ts.strftime("%B"),
                Department=ind["dept"],
                KPI_Category=ind["category"],
                Indicator=ind["name"],
                Unit=ind["unit"],
                Direction=ind["direction"],  # higher_is_better / lower_is_better
                Target=round(ind["target"], 2),
                Actual=round(float(actual), 2),
            ))
    df = pd.DataFrame(rows)
    df = df.sort_values(["Department", "Indicator", "Year", "Month_Num"]).reset_index(drop=True)
    return df


def achievement_rate(row):
    """% of target achieved, direction-adjusted. Thin wrapper around utils.achievement_rate."""
    return _achievement_rate(row["Actual"], row["Target"], row["Direction"])


# ---------------------------------------------------------------
# 2. CLIENT SATISFACTION SURVEY
# ---------------------------------------------------------------
QUESTIONS = [
    ("Access", "ACC1", "How would you rate the waiting time you experienced before receiving service?"),
    ("Access", "ACC2", "How satisfied are you with the facility's operating hours and appointment availability?"),
    ("Communication", "COM1", "How clearly did the health worker explain your diagnosis and treatment plan?"),
    ("Communication", "COM2", "Were you given enough opportunity to ask questions and raise concerns?"),
    ("Respect", "RES1", "How would you rate the courtesy and respect shown to you by staff?"),
    ("Respect", "RES2", "Did you feel your privacy and dignity were maintained during your visit?"),
    ("Information", "INF1", "Were you given clear information about your medication and how to use it?"),
    ("Information", "INF2", "Did you receive adequate information about follow-up care or referral, if needed?"),
]

# How strongly each domain's satisfaction is pulled by that department's
# monthly KPI achievement score (0 = no link, 1 = fully driven by it).
DOMAIN_KPI_WEIGHT = {"Access": 0.70, "Information": 0.40, "Communication": 0.30, "Respect": 0.10}

SURVEY_DEPTS = ["Emergency", "Pharmacy", "Laboratory", "Delivery Ward", "Outpatient"]
AGE_GROUPS = ["<18", "18-30", "31-45", "46-60", "60+"]
AGE_WEIGHTS = [0.06, 0.34, 0.32, 0.19, 0.09]


def build_survey_dataset(kpi_df):
    perf = department_monthly_performance(kpi_df)
    perf_lookup = {(r.Department, r.Year, r.Month_Num): r.Perf_Score for r in perf.itertuples()}

    rows = []
    respondent_id = 1
    for month_ts in MONTHS:
        year, mnum = month_ts.year, month_ts.month
        for dept in SURVEY_DEPTS:
            n_respondents = rng.integers(3, 9)  # 3-8 clients surveyed per dept per month
            perf_score = perf_lookup.get((dept, year, mnum), 85.0)
            # normalize performance score (typ. range ~50-115) to a -1..+1 pull factor
            pull = np.clip((perf_score - 85) / 35, -1.3, 1.3)

            for _ in range(n_respondents):
                age_group = rng.choice(AGE_GROUPS, p=AGE_WEIGHTS)
                sex = rng.choice(["Female", "Male"])
                visit_day = rng.integers(1, 28)
                survey_date = date(year, mnum, visit_day)

                for domain, qcode, qtext in QUESTIONS:
                    weight = DOMAIN_KPI_WEIGHT[domain]
                    # base satisfaction propensity + performance pull for this domain
                    score = 0.15 + weight * pull + rng.normal(0, 0.35)
                    # map continuous score -> 3-class response
                    if score > 0.25:
                        response = "Satisfied"
                    elif score < -0.25:
                        response = "Unsatisfied"
                    else:
                        response = "Neutral"

                    rows.append(dict(
                        Respondent_ID=f"R{respondent_id:05d}",
                        Survey_Date=survey_date.isoformat(),
                        Year=year, Month_Num=mnum,
                        Department=dept,
                        Age_Group=age_group,
                        Sex=sex,
                        Question_Domain=domain,
                        Question_Code=qcode,
                        Question_Text=qtext,
                        Satisfaction_Response=response,
                    ))
                respondent_id += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)

    kpi_df = build_kpi_dataset()
    kpi_path = os.path.join(out_dir, "ethiopia_health_kpis_monthly.csv")
    kpi_df.to_csv(kpi_path, index=False)
    print(f"Wrote {len(kpi_df):,} rows -> {kpi_path}")

    survey_df = build_survey_dataset(kpi_df)
    survey_path = os.path.join(out_dir, "client_satisfaction_survey.csv")
    survey_df.to_csv(survey_path, index=False)
    print(f"Wrote {len(survey_df):,} rows -> {survey_path}")
