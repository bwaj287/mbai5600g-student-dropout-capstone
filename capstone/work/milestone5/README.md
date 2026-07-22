# Milestone 5 Workspace

This folder contains the Milestone 5 model-development work.

## Current Working Goal

Milestone 4 established baseline models. Milestone 5 moves from baseline modeling into focused model improvement:

- tune selected model families
- compare tuned results against Milestone 4 baselines
- tune classification thresholds for early-warning tasks
- prepare model explanations for final candidate models

## Initial Modeling Focus

The first round focuses on the models that were strongest or most relevant in Milestone 4:

- `UCI multiclass reproduction`
  - tuned model: `XGBoost`
  - selection metric: `macro F1`
- `UCI binary early warning`
  - tuned models: `Logistic Regression` and `XGBoost`
  - selection metric: `F1`
  - added threshold tuning
- `OULAD binary early warning`
  - tuned model: `XGBoost`
  - selection metric: `F1`
  - added threshold tuning

## Files

- `milestone5_model_development.py`
  - Runs focused hyperparameter tuning
  - Compares tuned models with Milestone 4 baselines
  - Saves threshold curves for binary tasks
  - Saves confusion matrices and feature-importance plots
  - Generates SHAP explanation outputs for the tuned XGBoost models

- `analyze_model_explanations.py`
  - Groups SHAP results into shared feature families
  - Produces the explanation-shift comparison used for the novel contribution discussion

- `modeling_results_walkthrough.py`
  - Provides a simple course-notebook-style walkthrough of the Milestone 5 outputs
  - Uses direct loading, `info()`, `head()`, printed comparison tables, and short interpretations
  - Mirrors the style of the professor sample notebooks more closely than the longer modeling script

- `results/`
  - `tuned_model_comparison.csv`
  - `tuned_model_metrics.json`
  - feature-importance CSV files
  - SHAP summary CSV files
  - `shap_feature_family_comparison.csv`
  - `explanation_shift_summary.json`

- `figures/`
  - tuned confusion matrices
  - threshold tuning curves
  - feature-importance plots
  - SHAP plots
  - `shap_feature_family_comparison.png`

## Run Command

```bash
uv run --with pandas --with scikit-learn --with matplotlib --with xgboost --with shap python3 capstone/work/milestone5/milestone5_model_development.py
uv run --with pandas --with matplotlib python3 capstone/work/milestone5/analyze_model_explanations.py
uv run --with pandas python3 capstone/work/milestone5/modeling_results_walkthrough.py
```

The main modeling script imports `xgboost` and `shap` directly, so the run command above includes both packages.

## Official Milestone 5 Requirement Alignment

The official `Activity 5.pdf` frames this milestone as `Advanced Modeling & Optimization`.

The current work maps to the required sections as follows:

- `Novel Contribution Implementation`
  - cross-dataset robustness and explanation-shift analysis using UCI and OULAD
- `Advanced Model Development`
  - tuned Logistic Regression and XGBoost pipelines
- `Hyperparameter Optimization`
  - structured parameter grids and validation-set model selection
- `Comparative Model Evaluation`
  - baseline-vs-tuned comparison table and confusion matrices
- `Model Interpretation and Explainability`
  - feature importance, SHAP summaries, and feature-family SHAP comparison
- `Complexity vs. Performance Trade-off`
  - discussion that UCI benefits more from a simpler tuned Logistic Regression, while OULAD benefits from XGBoost

## Relationship To Milestone 4

Milestone 5 uses the model-ready datasets created in Milestone 4:

- `capstone/work/milestone4/data/uci_multiclass_model_ready.csv`
- `capstone/work/milestone4/data/uci_binary_early_model_ready.csv`
- `capstone/work/milestone4/data/oulad_binary_early_model_ready.csv`

The Milestone 4 baseline comparison is used as the reference point for improvement.

## First Run Results

The first Milestone 5 run completed focused tuning and SHAP outputs.

| Task | Tuned model | Baseline primary metric | Tuned test primary metric | Change |
| --- | --- | ---: | ---: | ---: |
| UCI multiclass reproduction | XGBoost | `0.704` macro F1 | `0.696` macro F1 | `-0.008` |
| UCI binary early warning | Logistic Regression | `0.783` F1 | `0.793` F1 | `+0.010` |
| UCI binary early warning | XGBoost | `0.783` F1 | `0.767` F1 | `-0.016` |
| OULAD binary early warning | XGBoost | `0.729` F1 | `0.738` F1 | `+0.009` |

## Early Interpretation

- The tuned `Logistic Regression` model is currently the strongest UCI early-warning candidate.
- The tuned `XGBoost` model is currently the strongest OULAD early-warning candidate.
- UCI multiclass XGBoost improved on validation but declined slightly on the test set, which suggests that additional tuning should be treated carefully.
- The OULAD tuned model selected a higher threshold (`0.60`), which improves practical classification balance for withdrawn-student detection.
- SHAP outputs were generated for the tuned XGBoost models and can support the later explanation-stability discussion.

## First SHAP Signals

- UCI multiclass SHAP is led by second-semester approved units, first-semester approved units, tuition status, grades, and age.
- UCI early-warning SHAP is led by first-semester approved units, tuition status, age, first-semester grade, and first-semester enrollment.
- OULAD early-warning SHAP is led by early assessment submission ratio, early assessment scores, weighted score ratio, studied credits, and selected course/module indicators.

## Feature-Family Explanation Shift

The feature-family comparison strengthens the project's novel contribution.

- UCI early-warning XGBoost:
  - `early_academic_progress`: `43.8%` of total mean absolute SHAP
  - `financial_support`: `20.4%`
  - `demographics`: `12.3%`
- OULAD early-warning XGBoost:
  - `early_academic_progress`: `54.5%`
  - `early_engagement`: `19.1%`
  - `program_setup`: `17.4%`

This shows that the two datasets share academic-progress signals, but OULAD adds a major behavioral engagement component that is not available in UCI.

## Current Takeaway

The first Milestone 5 run does not simply show that a more complex model is always better. Instead, it gives a stronger project story:

- Simple linear modeling remains strong for UCI early warning.
- Boosted trees remain useful for OULAD because the data is larger, multi-table, and more behaviorally complex.
- The feature signals differ across datasets, which directly supports the project focus on transferability and explanation stability.
