# Milestone 4 Workspace

This folder contains the Milestone 4 working artifacts for preprocessing and baseline modeling.

## What Was Completed

- Built a UCI multiclass baseline dataset for reproduction-oriented modeling.
- Built a UCI binary early-warning dataset by removing all second-semester variables.
- Built an OULAD student-level binary early-warning dataset using a fixed 75-day cutoff.
- Aggregated OULAD assessment activity and VLE behavior into model-ready student features.
- Exported a shared feature schema to support later cross-dataset work.
- Ran two baseline models for each task: logistic regression and random forest.

## Folder Layout

- `data/`
  - `uci_multiclass_model_ready.csv`
  - `uci_binary_early_model_ready.csv`
  - `oulad_binary_early_model_ready.csv`
  - `shared_feature_schema.json`
  - `shared_feature_schema.csv`
- `results/`
  - `prep_summary.json`
  - `baseline_metrics.json`
  - `baseline_comparison.csv`
  - feature-importance CSV files for each task/model
- `figures/`
  - confusion matrices for each task/model
  - top-feature plots for each task/model

## Modeling Setup

- Random seed: `42`
- UCI reproduction task:
  - target: original 3-class `Target`
- UCI early-warning task:
  - target: `is_attrition`
  - positive class: `Dropout`
  - second-semester variables removed to reduce timing leakage
- OULAD early-warning task:
  - target: `is_attrition`
  - positive class: `Withdrawn`
  - fixed cutoff: `75` days
  - `date_unregistration` excluded
  - early assessment and VLE events aggregated to student level

## Baseline Results

| Task | Best model | Main metrics |
| --- | --- | --- |
| UCI multiclass reproduction | Random forest | accuracy `0.766`, macro F1 `0.703` |
| UCI binary early warning | Logistic regression | accuracy `0.859`, F1 `0.785`, ROC AUC `0.913` |
| OULAD binary early warning | Logistic regression | accuracy `0.825`, F1 `0.732`, ROC AUC `0.879` |

Full comparison is in `results/baseline_comparison.csv`.

## Main Takeaways

- The UCI reproduction baseline is strongest when second-semester academic variables are allowed. The top drivers are approval and grade fields from the first and second semesters.
- The UCI early-warning version still performs well after removing second-semester fields, which is a better setup for intervention-oriented modeling.
- The OULAD 75-day model is weaker than UCI but still useful. Early assessment completion and score quality are among the strongest attrition signals.
- OULAD feature importance is still partly course-sensitive, especially `code_module` and `code_presentation`. That is acceptable for a baseline, but later validation should test more robust cross-course and cross-dataset generalization.

## Files To Review First

- `results/baseline_comparison.csv`
- `results/baseline_metrics.json`
- `figures/uci_binary_early_logistic_regression_top_features.png`
- `figures/oulad_binary_early_logistic_regression_top_features.png`
- `data/shared_feature_schema.json`

## Re-run Commands

```bash
uv run --with pandas python3 capstone/work/milestone4/run_milestone4_prep.py
uv run --with pandas --with scikit-learn --with matplotlib python3 capstone/work/milestone4/run_milestone4_baselines.py
```

## Caveats

- There is no standalone Milestone 4 requirements PDF in the repo, so this workspace was derived from the Milestone 3 deliverables and the course schedule file.
- The UCI dataset stores many categorical fields as numeric codes. The baseline script explicitly converts those coded fields back into categorical features before modeling.
- The OULAD baseline uses a fixed 75-day window to keep the setup simple and leakage-aware. Later milestones can test alternate time windows and stronger validation strategies.
