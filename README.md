# AI-Based KPI Monitoring & PMT Decision Support System

An end-to-end, deployable prototype for a health-center Performance Monitoring
Team (PMT): it **predicts** next-month KPI performance, **detects and grades**
Target-vs-Actual gaps, **recommends** a concrete action plan for each gap, and
**explains** what is driving client (dis)satisfaction — all from two linked
datasets (facility KPIs + client-satisfaction survey).

## Why this design (for judges)

Most "AI dashboard" hackathon entries stop at descriptive charts. This one
covers all four analytics layers a PMT actually needs, and is honest about
where the ML adds value vs. where a transparent rule survives scrutiny better
in a health/government setting:

| Layer | What it does | Where |
|---|---|---|
| **Descriptive** | Target vs Actual, achievement %, trend charts | `app.py` |
| **Diagnostic** | Why is satisfaction low? Correlation + ML feature importance linking survey responses back to KPI performance | `src/satisfaction_analysis.py` |
| **Predictive** | Next-month forecast per indicator, auto-selecting the best of 3 candidate models per series | `src/forecasting.py` |
| **Prescriptive** | Gap severity grading + an explainable recommendation engine (knowledge base + TF-IDF fallback for unseen indicators) | `src/gap_recommendation.py` |

The two datasets are **deliberately linked**: synthetic survey responses are
generated so that "Access"-domain satisfaction tracks that department's real
KPI performance far more strongly than "Respect" does. The driver-analysis
model successfully re-discovers this from the raw data (see the printed
correlation table when you run `src/satisfaction_analysis.py`) — a concrete,
demoable proof that the system finds real signal, not just displays numbers.

## Project structure

```
kpi_pmt_system/
├── app.py                          # Streamlit dashboard (the deployable app)
├── requirements.txt
├── data/
│   ├── ethiopia_health_kpis_monthly.csv       # 648 rows: 18 indicators x 36 months
│   └── client_satisfaction_survey.csv         # ~7,900 rows: 8 questions x respondents
└── src/
    ├── utils.py                    # shared achievement-rate / severity / perf-score logic
    ├── generate_data.py            # synthetic data generator (reproducible, seed=42)
    ├── forecasting.py              # per-indicator model selection + next-month forecast
    ├── gap_recommendation.py       # severity grading + recommendation engine
    └── satisfaction_analysis.py    # satisfaction scoring + ML driver analysis
```

## Datasets

### `ethiopia_health_kpis_monthly.csv`
Monthly Target vs Actual for **18 indicators** across **6 department/service
areas** (Emergency, Pharmacy, Laboratory, Delivery Ward, Outpatient,
Facility-Wide), Jan 2023 - Dec 2025.

Columns: `Year, Month_Num, Month_Name, Department, KPI_Category, Indicator,
Unit, Direction, Target, Actual`

`Direction` is `higher` (bigger is better, e.g. immunization coverage) or
`lower` (smaller is better, e.g. waiting time) — this is what lets one
formula (`utils.achievement_rate`) score every indicator type consistently.

### `client_satisfaction_survey.csv`
Client-satisfaction responses (`Satisfied` / `Neutral` / `Unsatisfied`) across
**8 questions in 4 domains** (Access, Communication, Respect, Information),
mirroring a standard 4-section client-satisfaction tool.

Columns: `Respondent_ID, Survey_Date, Year, Month_Num, Department, Age_Group,
Sex, Question_Domain, Question_Code, Question_Text, Satisfaction_Response`

**Both datasets are synthetic** (seeded, reproducible via
`python src/generate_data.py`), built to be realistic for a demo. To go live:

- Replace `ethiopia_health_kpis_monthly.csv` with a DHIS2 pivot-table export
  (Data Element x Period), reshaped to the same columns.
- Replace `client_satisfaction_survey.csv` with a KoboToolbox export (one row
  per question response) — the earlier Goro HC client-satisfaction XLSForm
  already collects Access/Communication/Respect/Information responses, so
  only the response-scale mapping and column renaming would be needed.

No other file needs to change — every model reads the same column names.

## The models, briefly

**Forecasting (`forecasting.py`)** — With only ~36 monthly points per
indicator, a heavy model would overfit. Instead, for every indicator the
system fits three candidates — naive persistence, a linear trend+seasonal
(Fourier) model, and a Random Forest on lag/rolling features — evaluates each
on a 6-month walk-forward holdout, and keeps whichever generalizes best *for
that specific indicator*. This is the "best, feasible" choice for a
small, heterogeneous multi-KPI dataset, and it's transparent: the dashboard
shows which model won and its holdout error for every indicator.

**Gap severity & recommendations (`gap_recommendation.py`)** — Achievement
rate is banded into On Track / Minor / Moderate / Critical. Each indicator
maps to a curated reason + action-plan template (the same style as the
mockup's "AI Recommendation" column); severity scales the urgency framing
("URGENT —" vs "Monitor & improve —"). If a new indicator not in the
knowledge base ever shows up (e.g. after swapping in real DHIS2 data), a
TF-IDF similarity match finds the closest known indicator and generalizes its
template rather than failing outright.

**Satisfaction driver analysis (`satisfaction_analysis.py`)** — Converts
responses to a 5-point score (matching the mockup's "X.X / 5" card), then
runs two complementary, cross-checking analyses: a simple domain-vs-KPI
correlation (transparent, easy to defend to a non-technical manager) and a
Random Forest feature-importance ranking (Department, Question_Domain,
Age_Group, Sex, department-month KPI performance → predicted response). Both
are shown side by side rather than picking one "true" ranking.

## Run locally

```bash
pip install -r requirements.txt
python src/generate_data.py       # regenerate the CSVs (optional, already included)
streamlit run app.py
```

## Deploy (Streamlit Community Cloud — free, ~5 minutes)

1. Push this folder to a public (or private) GitHub repo.
2. Go to https://share.streamlit.io -> "New app".
3. Pick the repo/branch, set **Main file path** to `app.py`, deploy.
4. Streamlit Cloud installs `requirements.txt` automatically. No secrets or
   API keys are needed since the CSVs ship with the repo.

(Any host that runs a `streamlit run app.py` process — Render, Hugging Face
Spaces, an internal server — works the same way.)

## Limitations & honest next steps (worth saying out loud in the pitch)

- Both datasets are synthetic; results demonstrate the *pipeline*, not real
  Goro HC performance. Swapping in real DHIS2/Kobo exports is a data-mapping
  exercise, not a code change.
- The satisfaction classifier's per-response accuracy (~40% on 3 classes,
  printed by `satisfaction_analysis.py`) is modest — it's tuned for
  interpretability (driver analysis) rather than for predicting one person's
  answer, and the pitch should say so rather than oversell it.
- Recommendation text is templated, not generative — deliberately, for
  auditability in a health/government setting. A natural next step is
  logging which recommendations were acted on and their outcome, to move
  from rule-based to outcome-informed recommendations over time.
- Real Ethiopian PMT cycles run on the Ethiopian fiscal year (Hamle-Sene);
  the dataset uses Gregorian months for simplicity — adding an Ethiopian
  calendar column is a small extension if the judges' rubric expects it.

## Suggested 60-second pitch structure

1. **Problem**: Health centers set targets (HMIS/DHIS2, WBP) but reviewing
   gaps and deciding what to do about them is manual and after-the-fact.
2. **What it does**: One system that predicts next month's KPI performance,
   grades every gap by severity, tells the manager *why* it's happening and
   *what to do*, and links it to what patients actually experience.
3. **The proof point**: Show the driver-analysis correlation table — the
   model independently rediscovers that Access-related satisfaction tracks
   operational performance, without being told to.
4. **Feasibility**: Runs on 36 months of monthly data per indicator — no big
   -data requirement, deployable today on Streamlit Cloud for free, and
   built to plug into DHIS2/KoboToolbox exports already in use at the
   facility level.
