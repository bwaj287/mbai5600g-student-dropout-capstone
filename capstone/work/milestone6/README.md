# Milestone 6 Workspace

This folder contains the Milestone 6 validation and business-impact work.

## Current Goal

Milestone 6 validates the optimized early-warning models from Milestone 5 and translates the technical results into practical deployment and business-impact conclusions.

## What We Did

- Validated the final selected UCI and OULAD early-warning models on the held-out test sets.
- Checked robustness with repeated stratified hold-out splits.
- Tested sensitivity to simulated `10%` missing feature values and `5%` numeric noise.
- Analyzed false positives, false negatives, and feature patterns behind common errors.
- Reviewed segment-level performance as an initial fairness and reliability screen.
- Converted model results into operational impact estimates per `1,000` students.
- Drafted the Milestone 6 report with validation, business impact, deployment feasibility, ethics, and final recommendations.

## Final Candidate Models

| Setting | Final model | Key validation result |
| --- | --- | --- |
| UCI student-record early warning | Logistic Regression | F1 `0.793`, recall `0.820`, ROC AUC `0.911` |
| OULAD online-learning early warning | XGBoost | F1 `0.738`, recall `0.719`, ROC AUC `0.885` |

## Main Outputs

- `Milestone6.pdf`
  - final PDF report for submission/GitHub
- `Milestone6.docx`
  - local Word version, ignored by git
- `milestone6_validation_impact.py`
  - runs validation, robustness checks, error analysis, segment analysis, and business-impact calculations
- `results/`
  - validation, robustness, error, segment, and business-impact CSV outputs
- `figures/`
  - report-ready validation, robustness, error, and business-impact figures

## Key Findings

- The UCI Logistic Regression model is the strongest structured-record early-warning candidate.
- The OULAD XGBoost model is the stronger candidate for online-learning behavior data.
- Both models are stable under small numeric noise.
- Simulated missingness is the biggest robustness concern and should be monitored in deployment.
- Operationally, the UCI model flags about `344` students per `1,000` and finds about `263` true at-risk students.
- The OULAD model flags about `295` students per `1,000` and finds about `224` true at-risk students.
- The final recommendation is cautious pilot deployment as advisor decision support, not automated decision-making.

## Run Commands

```bash
uv run --with pandas --with scikit-learn --with matplotlib --with xgboost --with shap python3 capstone/work/milestone6/milestone6_validation_impact.py
```

The report builder is kept as a local generated-artifact helper:

```bash
python3 capstone/work/milestone6/build_milestone6_report.py
```

## Relationship To Earlier Milestones

Milestone 6 uses the model-ready datasets from Milestone 4 and the tuned model settings from Milestone 5. It does not introduce a new algorithm family. The purpose is validation, reliability assessment, business interpretation, deployment feasibility, and risk assessment.
