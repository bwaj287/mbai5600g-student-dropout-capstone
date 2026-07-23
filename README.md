# Student Dropout Early Warning Capstone

This repository records our MBAI 5600G capstone work on early identification of
students who may leave a course or program. The project uses the UCI student
dropout dataset and the Open University Learning Analytics Dataset (OULAD).

## Research Design

The project developed in four parts:

1. Reproduce the main model family from Islam et al. (2025) on the UCI
   multiclass outcome.
2. Build binary early-warning models for UCI and OULAD.
3. Compare the important feature families in the two datasets with SHAP.
4. Test how OULAD performance changes when only the first 35, 60, or 75 days
   of activity are available.

UCI and OULAD do not share the same raw columns or the same outcome definition.
For that reason, our comparison is at the feature-family and model-behavior
level. It is not a claim that one fitted model transfers directly between the
two datasets.

## Milestones

| Milestone | Work completed |
| --- | --- |
| 1 | Proposed the early-warning problem and initial project scope |
| 2 | Reviewed the literature and selected UCI and OULAD |
| 3 | Collected the data and completed exploratory data analysis |
| 4 | Prepared model-ready data and trained baseline models |
| 5 | Tuned selected models and produced SHAP explanations |
| 5 extension | Compared OULAD models at days 35, 60, and 75 |
| 6 | Completed validation, error analysis, robustness tests, and impact estimates |
| 7 | Prepared the final presentation |

## Main Results

| Task | Selected model | Test result |
| --- | --- | --- |
| UCI multiclass reproduction | XGBoost baseline | Accuracy `0.768`, macro F1 `0.704` |
| UCI binary early warning | Tuned Logistic Regression | F1 `0.793`, ROC AUC `0.911` |
| OULAD binary early warning | Tuned XGBoost | F1 `0.738`, ROC AUC `0.885` |

The temporal OULAD experiment produced F1 scores of `0.682`, `0.716`, and
`0.738` at days 35, 60, and 75. More activity history improved prediction, but
later alerts also leave less time for intervention.

## Repository Layout

```text
capstone/
  material/                  course reference material
  work/
    milestone1/
    milestone2/
    milestone3/
    milestone4/
    milestone5/
    milestone5extend/
    milestone6/
    milestone7/
```

Each milestone folder has a short README, the submitted PDF, and the code or
results used at that stage. Word drafts, report-building scripts, meeting
transcripts, rendered previews, and superseded slide decks stay local.

## Reproducing The Analysis

Run the files in milestone order:

```bash
python3 capstone/work/milestone3/code/milestone3_eda_pipeline.py
python3 capstone/work/milestone4/run_milestone4_prep.py
python3 capstone/work/milestone4/run_milestone4_baselines.py
python3 capstone/work/milestone5/milestone5_model_development.py
python3 capstone/work/milestone5/analyze_model_explanations.py
python3 capstone/work/milestone5extend/temporal_early_warning_analysis.py
python3 capstone/work/milestone6/milestone6_validation_impact.py
```

The main packages are pandas, NumPy, Matplotlib, scikit-learn, XGBoost, and
SHAP. The scripts use fixed random seeds so the saved tables can be reproduced.

## Data Note

The UCI source file and the OULAD source archive are under
`capstone/work/milestone3/data/raw/`. The extracted OULAD `studentVle.csv` is
not tracked because it is about 433 MB. The scripts can read that table from
`oulad.zip` instead.
