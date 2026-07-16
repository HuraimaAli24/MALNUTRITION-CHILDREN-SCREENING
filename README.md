# Child Malnutrition Risk Screening — Multi-Task Deep Learning with Uncertainty and Explainability

A real-time deep learning system that predicts a child's risk of stunting, wasting, and
underweight from household and demographic factors, built on DHS (Demographic and Health
Survey) data and WHO Child Growth Standards.

## Motivation

Community health workers often need to identify at-risk children before formal anthropometric
screening is possible (e.g., no scale/height board on hand, or to prioritize which households to
visit first). This project asks: **can a model flag malnutrition risk using only the household
and demographic information a health worker already has** — age, sex, household wealth, region,
residence type, and mother's education/age — rather than requiring a direct measurement?

This is intentionally a harder and more useful problem than predicting malnutrition from other
body measurements (which would be a much easier, but far less practically useful, task).

## Approach

### 1. Data
- DHS Children's Recode (KR) survey data, which includes anthropometric measurements and
  pre-computed WHO Child Growth Standards z-scores (`hw70`/`hw71`/`hw72`) for height-for-age
  (stunting), weight-for-age (underweight), and weight-for-height (wasting).
- Standard DHS cleaning applied: flagged/implausible z-scores removed (values coded 9998/9999
  or |z| > 6 SD), records without valid anthropometry (unmeasured children) excluded, restricted
  to children under 5 years.

### 2. Multi-task architecture
Rather than three independent models, a single shared-trunk neural network learns a joint
representation, with three task-specific heads (stunting, wasting, underweight). This reflects
the real clinical correlation between these conditions — a malnourished child is often at
elevated risk across all three indicators simultaneously. Each head outputs:
- a continuous z-score regression (matching the WHO Child Growth Standards scale), and
- a 3-class severity classification (normal / moderate / severe), trained with inverse-frequency
  class weighting since severe cases are the rare minority class and the costliest to miss.

### 3. Uncertainty quantification
Monte Carlo Dropout (dropout kept active at inference, ~30 stochastic forward passes) produces
a mean and standard deviation for every prediction, so the system communicates when it is
uncertain rather than presenting a single number as fact — important for a decision-support
tool in a health context.

### 4. Explainability
SHAP (KernelExplainer) attributes each prediction to the input features that drove it (e.g.,
"this child's elevated stunting risk is mainly driven by region and household wealth"), making
the model's reasoning interpretable to non-technical users.

### 5. Real-time serving
- **FastAPI** backend (`/predict` endpoint), ~20–40ms inference latency including MC Dropout and
  SHAP explanation.
- **Streamlit** frontend: enter a child's information, get an instant risk breakdown with
  confidence intervals and a feature-contribution chart.

## Design decisions worth noting

- **Severity is derived from the regression z-score (WHO-standard thresholds), not the
  classifier head.** During development, the classifier head — trained with heavy class
  weighting to catch rare severe cases — sometimes disagreed with the regression output. Rather
  than hide this, the system displays severity from the clinically standard z-score threshold
  (always consistent with the reported number) and surfaces classifier disagreement as an
  explicit cross-check signal, flagging cases where the two heads disagree.
- **Class-weighted loss trades precision for recall on severe cases deliberately.** In a
  screening context, a false positive costs a follow-up checkup; a false negative on a severely
  malnourished child costs far more. Severe-case recall improved from near-zero to 53–70% after
  weighting, at the cost of more false positives — the right tradeoff for this use case.

## Results (test set, n=311)

| Indicator | Regression MAE (SD) | Severe-class recall |
|---|---|---|
| Stunting (HAZ) | 1.44 | 0.69 |
| Underweight (WAZ) | 0.99 | 0.53 |
| Wasting (WHZ) | 1.22 | 0.70 |

## Limitations

- Trained on a DHS **Model Dataset** (synthetic/practice data) during initial development;
  intended to be retrained on real country-level DHS data (India / Afghanistan, pending access
  approval) before being presented as reflecting real-world patterns.
- Predicting from demographic/household proxies alone is inherently harder than using direct
  measurements — regression error (~1–1.4 SD) reflects this genuine difficulty, not a modeling
  flaw. This is a screening/triage aid, not a diagnostic tool.
- Regional and household-level factors captured here don't include some clinically important
  variables (e.g., recent illness, breastfeeding practices, dietary diversity) that DHS collects
  elsewhere and could improve the model in future work.
- This tool does not replace clinical assessment or physical measurement.

## Project structure
```
malnutrition_project/
├── data/               # DHS raw + cleaned data
├── src/                # data_prep.py, model.py, train.py, inference.py
├── models/             # trained weights, scaler, encoders, metrics
├── app/                # api.py (FastAPI), frontend.py (Streamlit)
└── README.md
```

## Running it

```bash
pip install -r requirements.txt

# 1. Clean data
python src/data_prep.py

# 2. Train model
python src/train.py

# 3. Run API
cd app && uvicorn api:app --reload --port 8000

# 4. Run frontend (separate terminal)
cd app && streamlit run frontend.py
```

## Ethical considerations

Child health data is sensitive. This project uses de-identified survey data (no names, no
household identifiers beyond anonymized codes) and is intended strictly as a research/portfolio
project demonstrating applied ML methodology for humanitarian health screening, not as a
deployed clinical system.
