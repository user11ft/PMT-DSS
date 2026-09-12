"""
gap_recommendation.py
----------------------
The "prescriptive" layer of the PMT system: turns a Target-vs-Actual
gap into (a) a plain-language reason and (b) a concrete action plan,
exactly like the "Gap Analysis & Recommendations" table in the
dashboard mockup.

Design (why this is explainable, not a black box -- important in a
health/government setting where managers must be able to justify a
recommendation to auditors):
  1. A curated knowledge base gives an exact, expert-reviewed
     reason + action template for every indicator we know about.
  2. For an indicator NOT in the knowledge base (e.g. you plug in a
     new/real DHIS2 indicator later), a TF-IDF text-similarity
     fallback finds the closest known indicator (by name + KPI
     category) and reuses/generalizes its action template, so the
     system degrades gracefully instead of failing.
  3. Severity (from utils.classify_severity) scales the *urgency
     framing* of the same action list, so the same underlying issue
     reads as routine monitoring when Minor and an urgent directive
     when Critical.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import achievement_rate, classify_severity
from nlp_recommendation import compute_trend_features, generate_dynamic_recommendation

# ---------------------------------------------------------------
# Knowledge base: Indicator -> (reason for gap, recommended actions)
# Mirrors the style of the hackathon mockup's "AI Recommendation" column.
# ---------------------------------------------------------------
KNOWLEDGE_BASE = {
    "Avg_Triage_to_Treatment_Time_Min": dict(
        reason="Rising patient inflow against limited triage bay space is lengthening wait times.",
        action="Increase staffing during peak hours; add 1-2 triage beds; introduce fast-track lanes for minor cases."),
    "Emergency_Case_Fatality_Rate_Pct": dict(
        reason="Delays in triage-to-treatment and possible protocol gaps are affecting emergency outcomes.",
        action="Audit recent case-fatality records; refresh emergency triage protocol training; ensure crash-cart/equipment readiness checks."),
    "Emergency_Bed_Availability_Rate_Pct": dict(
        reason="Growing admissions are outpacing available emergency bed capacity.",
        action="Expand emergency bed capacity where feasible; speed up patient disposition/discharge decisions; set a clear step-down transfer protocol to the wards."),

    "Essential_Medicine_Availability_Rate_Pct": dict(
        reason="Supply chain delays are reducing on-shelf availability of essential medicines.",
        action="Improve inventory monitoring frequency; expedite supplier orders; pre-position buffer stock for high-use items."),
    "Stockout_Rate_Pct": dict(
        reason="Pharmacy stock levels are falling below the reorder threshold before resupply arrives.",
        action="Set/enforce a minimum stock re-order level; shorten the procurement lead time; track fast-moving items with a dedicated stock-alert list."),
    "Avg_Prescription_Fill_Time_Min": dict(
        reason="Manual dispensing steps and queue volume are slowing prescription turnaround.",
        action="Introduce a prescription pre-sorting/priority queue; cross-train additional dispensers during peak hours."),

    "Avg_Test_Turnaround_Time_Hours": dict(
        reason="High sample volume against limited equipment/staff capacity is slowing lab turnaround.",
        action="Add automation where feasible; redistribute workload across shifts; cross-train staff to cover peak sample-volume periods."),
    "Lab_QC_Pass_Rate_Pct": dict(
        reason="Quality-control pass rates are drifting, suggesting equipment calibration or reagent-handling issues.",
        action="Schedule equipment calibration/maintenance; refresh SOP training on sample handling; audit a sample of recent QC failures."),
    "Sample_Rejection_Rate_Pct": dict(
        reason="Pre-analytical errors (labeling, collection technique, transport delays) are causing sample rejections.",
        action="Retrain sample-collection staff on labeling/transport SOPs; add a pre-submission checklist at point of collection."),

    "Institutional_Delivery_Rate_Pct": dict(
        reason="Community awareness or access barriers may still be routing some deliveries outside the facility.",
        action="Strengthen community health worker referral linkages; run demand-generation sessions with kebele leaders; monitor referral pathway completion."),
    "Skilled_Birth_Attendance_Rate_Pct": dict(
        reason="Staffing coverage gaps on some shifts may leave deliveries without a skilled attendant present.",
        action="Review shift-coverage rosters for 24/7 skilled-attendant presence; fast-track any pending midwifery staffing requests."),
    "Delivery_Bed_Availability_Rate_Pct": dict(
        reason="Increased admissions combined with limited ward capacity and maintenance downtime are reducing bed availability.",
        action="Expand delivery/postnatal bed capacity where feasible; optimize early (safe) discharge and postnatal follow-up; coordinate stable-case referrals to decompress the ward."),

    "Total_OPD_Visits": dict(
        reason="Visit volume is trending above the planned staffing/space assumptions used to set this target.",
        action="Re-baseline the visit target using recent trend data; review outpatient staffing and room allocation against the new demand level."),
    "ANC4_Coverage_Rate_Pct": dict(
        reason="Clients are not returning for all four recommended antenatal visits, often due to distance or awareness gaps.",
        action="Use HEWs for ANC follow-up reminders/tracing; align ANC visit scheduling with market/community days to ease access."),
    "Full_Immunization_Coverage_Pct": dict(
        reason="Drop-off between vaccination doses suggests defaulter tracing or outreach gaps.",
        action="Strengthen defaulter tracing using the immunization register; schedule outreach sessions in under-covered kebeles."),
    "Avg_Outpatient_Waiting_Time_Min": dict(
        reason="Registration/triage throughput has not kept pace with growing outpatient volume.",
        action="Add a second registration point at peak hours; implement queue-ticketing with appointment slots to spread arrivals."),

    "HMIS_Reporting_Completeness_Pct": dict(
        reason="Some reporting units are submitting incomplete HMIS/DHIS2 monthly returns.",
        action="Send targeted completeness reminders to lagging reporting units; run a brief refresher on the reporting checklist."),
    "HMIS_Reporting_Timeliness_Pct": dict(
        reason="Monthly HMIS/DHIS2 submissions are arriving after the reporting deadline.",
        action="Set an internal submission deadline a few days ahead of the official one; assign a backup data-clerk for reporting-week coverage."),
}

# Severity -> urgency framing prefix applied to the action text.
SEVERITY_PREFIX = {
    "Critical Gap": "URGENT — ",
    "Moderate Gap": "Recommended — ",
    "Minor Gap": "Monitor & improve — ",
    "On Track": "",
}

_KB_INDICATORS = list(KNOWLEDGE_BASE.keys())


def _fallback_via_similarity(indicator_name: str, category: str, kb_categories: dict):
    """TF-IDF nearest-neighbor match for an indicator not in the knowledge base."""
    corpus = [f"{name.replace('_', ' ')} {kb_categories.get(name, '')}" for name in _KB_INDICATORS]
    query = f"{indicator_name.replace('_', ' ')} {category}"
    vec = TfidfVectorizer().fit(corpus + [query])
    matrix = vec.transform(corpus + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_idx = sims.argmax()
    best_name = _KB_INDICATORS[best_idx]
    return best_name, float(sims[best_idx])


def build_gap_report(kpi_df: pd.DataFrame, year: int = None, month_num: int = None,
                      use_nlp: bool = True) -> pd.DataFrame:
    """
    Produces the Gap Analysis & Recommendations table for one month
    (defaults to the latest month present in kpi_df).

    use_nlp=True (default) routes every row through the flexible NLP
    recommendation layer (nlp_recommendation.py): the reason/action text
    is composed per-row from the gap's own trend, streak and volatility,
    so two gaps on the same indicator no longer get identical wording
    just because they share a severity band. Set False to fall back to
    the original static "one sentence per indicator+severity" behavior.
    """
    df = kpi_df.copy()
    if year is None or month_num is None:
        latest = df.sort_values(["Year", "Month_Num"]).iloc[-1]
        year, month_num = int(latest["Year"]), int(latest["Month_Num"])

    kb_categories = {}  # indicator -> KPI_Category, for the similarity fallback text
    for _, r in df.drop_duplicates("Indicator").iterrows():
        kb_categories[r["Indicator"]] = r["KPI_Category"]

    if use_nlp:
        trend_df = compute_trend_features(df)
        snapshot = trend_df[(trend_df["Year"] == year) & (trend_df["Month_Num"] == month_num)].copy()
    else:
        snapshot = df[(df["Year"] == year) & (df["Month_Num"] == month_num)].copy()

    rows = []
    for _, r in snapshot.iterrows():
        if use_nlp:
            rate, severity = r["Achievement_Rate"], r["Gap_Severity"]
        else:
            rate = achievement_rate(r["Actual"], r["Target"], r["Direction"])
            severity = classify_severity(rate)

        if r["Indicator"] in KNOWLEDGE_BASE:
            kb = KNOWLEDGE_BASE[r["Indicator"]]
            match_type, match_conf = "exact", 1.0
        else:
            match_name, match_conf = _fallback_via_similarity(r["Indicator"], r["KPI_Category"], kb_categories)
            kb = KNOWLEDGE_BASE[match_name]
            match_type = f"generalized from '{match_name}'"

        row_out = dict(
            Year=year, Month_Num=month_num,
            Department=r["Department"], Indicator=r["Indicator"], KPI_Category=r["KPI_Category"],
            Target=r["Target"], Actual=r["Actual"], Unit=r["Unit"],
            Achievement_Rate=rate, Gap_Severity=severity,
            Recommendation_Source=match_type, Match_Confidence=round(match_conf, 2),
        )

        if use_nlp:
            gen = generate_dynamic_recommendation(r.to_dict(), kb["reason"], kb["action"])
            row_out.update(
                Reason_For_Gap=gen["reason"],
                Action_Plan_Recommendation=gen["action"],
                Trend=gen["trend"],
                Consecutive_Months_In_Gap=gen["streak"],
                Urgency_Score=gen["urgency_score"],
            )
        else:
            if severity == "On Track":
                reason = "Performance is meeting or exceeding target."
                action = "Maintain current practices; continue monthly monitoring to sustain performance."
            else:
                reason = kb["reason"]
                action = SEVERITY_PREFIX[severity] + kb["action"]
            row_out.update(Reason_For_Gap=reason, Action_Plan_Recommendation=action)

        rows.append(row_out)

    report = pd.DataFrame(rows)
    if use_nlp:
        # Rank by the blended Urgency_Score first (captures trend + streak + volatility,
        # not just the static severity band), tie-broken by achievement rate.
        return report.sort_values(["Urgency_Score", "Achievement_Rate"],
                                   ascending=[False, True]).reset_index(drop=True)

    severity_order = {"Critical Gap": 0, "Moderate Gap": 1, "Minor Gap": 2, "On Track": 3}
    report["_order"] = report["Gap_Severity"].map(severity_order)
    return report.sort_values(["_order", "Achievement_Rate"]).drop(columns="_order").reset_index(drop=True)


if __name__ == "__main__":
    kpi_df = pd.read_csv("../data/ethiopia_health_kpis_monthly.csv") \
        if __package__ else pd.read_csv("data/ethiopia_health_kpis_monthly.csv")
    report = build_gap_report(kpi_df)
    pd.set_option("display.width", 200)
    print(report[["Department", "Indicator", "Gap_Severity", "Trend", "Consecutive_Months_In_Gap",
                   "Urgency_Score", "Achievement_Rate", "Reason_For_Gap", "Action_Plan_Recommendation"]]
          .to_string(index=False))
