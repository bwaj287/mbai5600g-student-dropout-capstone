# Milestone 4 Workspace

This folder contains the Milestone 4 working artifacts for preprocessing and baseline modeling.

## Milestone 4 Goal

Milestone 4 was the transition from data understanding into actual modeling.

The practical goal was:

- turn the collected datasets into model-ready tables
- control obvious leakage risks
- run baseline models that can support later cross-dataset work

## What We Did Step By Step

### Step 1: Defined the modeling tasks

- `UCI multiclass reproduction`
  - original target: `Graduate`, `Dropout`, `Enrolled`
- `UCI binary early warning`
  - positive class: `Dropout`
- `OULAD binary early warning`
  - positive class: `Withdrawn`

### Step 2: Prepared the UCI datasets

- Built a multiclass version for reproduction-oriented benchmarking.
- Built an early-warning binary version by removing all second-semester variables.
- Treated coded categorical fields as categorical during modeling instead of leaving them as ordinary continuous numbers.

### Step 3: Prepared the OULAD dataset

- Worked from the multi-table OULAD structure instead of a single flat table.
- Used `code_module`, `code_presentation`, and `id_student` as the main join keys.
- Merged student-level information from:
  - `studentInfo`
  - `studentRegistration`
  - `studentAssessment`
  - `studentVle`
  - supporting metadata tables

### Step 4: Aggregated early-course features

- Used a fixed `75`-day early-warning cutoff.
- Aggregated assessment behavior into features such as:
  - early submission count
  - early score summaries
  - weighted-score summaries
- Aggregated VLE behavior into features such as:
  - total clicks
  - event counts
  - active days
  - unique site visits
  - activity-type click totals

### Step 5: Controlled leakage

- Removed `date_unregistration` from OULAD modeling inputs.
- Limited OULAD behavior features to the early time window.
- Removed UCI second-semester variables from the early-warning version.

### Step 6: Created a shared feature schema

- Exported `shared_feature_schema.json` and `shared_feature_schema.csv`.
- This groups fields into common concept families such as:
  - demographics
  - prior preparation
  - program setup
  - early academic progress
  - early engagement
- This matters because UCI and OULAD do not have matching raw columns, but they can still be compared at the concept level later.

### Step 7: Ran baseline models

- Ran `logistic regression` and `random forest` for each task.
- Exported:
  - metrics
  - confusion matrices
  - top-feature charts
  - feature-importance CSV files

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

## How To Read The Metrics

### F1

`F1` balances precision and recall.

- `precision`: when the model flags a student as high risk, how often is that correct
- `recall`: of the students who truly are high risk, how many does the model catch

This matters because an early-warning system should not miss too many at-risk students, but it also should not flood advisors with false alarms.

### ROC AUC

`ROC AUC` measures overall ranking and separation ability.

- It tells us how well the model gives higher risk scores to true attrition cases than to non-attrition cases.
- It is less tied to one fixed cutoff than F1 is.

For this project:

- `F1` is more about practical classification quality
- `ROC AUC` is more about overall discrimination quality

## Why Shared Feature Schema Matters

This is one of the most important design decisions in the project.

- `UCI` and `OULAD` do not have the same raw schema.
- We therefore cannot compare them column-by-column in a naive way.
- Instead, we align them by feature families.

Example:

- `UCI` has structured academic and administrative signals.
- `OULAD` has structured student information plus behavioral VLE signals.
- Both can still contribute to broader concepts like demographics, prior preparation, and early academic progress.

This shared schema is the bridge that later supports:

- cross-dataset generalization analysis
- explanation stability analysis
- comparisons that are conceptually fair rather than tied to raw field names

## Main Takeaways

- The UCI reproduction baseline is strongest when second-semester academic variables are allowed. The top drivers are approval and grade fields from the first and second semesters.
- The UCI early-warning version still performs well after removing second-semester fields, which is a better setup for intervention-oriented modeling.
- The OULAD 75-day model is weaker than UCI but still useful. Early assessment completion and score quality are among the strongest attrition signals.
- OULAD feature importance is still partly course-sensitive, especially `code_module` and `code_presentation`. That is acceptable for a baseline, but later validation should test more robust cross-course and cross-dataset generalization.

## What Milestone 4 Did Not Finish Yet

- It did not complete advanced modeling or hyperparameter optimization.
- It did not yet perform the full cross-dataset transfer experiments.
- It did not yet complete explanation-stability analysis across datasets.

Milestone 4 established the baseline and the modeling-ready data foundation. The later milestones are where the project becomes a stronger generalization study.

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

## Meeting Summary

If you need a short description for a meeting, Milestone 4 can be summarized like this:

- We converted both datasets into model-ready forms.
- We created an early-warning version of UCI and a student-level early-warning version of OULAD.
- We controlled the most obvious leakage risks.
- We ran baseline models and confirmed that both datasets now support the next stage of cross-dataset analysis.

## Caveats

- There is no standalone Milestone 4 requirements PDF in the repo, so this workspace was derived from the Milestone 3 deliverables and the course schedule file.
- The UCI dataset stores many categorical fields as numeric codes. The baseline script explicitly converts those coded fields back into categorical features before modeling.
- The OULAD baseline uses a fixed 75-day window to keep the setup simple and leakage-aware. Later milestones can test alternate time windows and stronger validation strategies.
