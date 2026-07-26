# Student Dropout Early Warning Capstone

This repository records our MBAI 5600G capstone work on early identification of
students who may leave a course or program. The project uses the UCI student
dropout dataset and the Open University Learning Analytics Dataset (OULAD).

## Research Design

The project developed in four parts:

1. Reproduce the main model family from Islam et al. (2025) on the UCI
   multiclass outcome.
2. Audit outcome timing and student overlap in the preliminary pipelines.
3. Build OULAD landmark models for students still enrolled at days 35, 60,
   and 75.
4. Compare operational landmark performance with a controlled same-record
   temporal experiment.
5. Test calibration, student-cluster uncertainty, SHAP feature-family shifts,
   and a later 2014J presentation holdout.

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
| Final report | Rebuilt the study with landmark outcomes, student-level splits, calibration, clustered uncertainty, and a later-presentation holdout |

## Main Results

| Task | Selected model | Test result |
| --- | --- | --- |
| UCI multiclass reproduction | XGBoost baseline | Accuracy `0.768`, macro F1 `0.704` |
| UCI enrolment-time benchmark | XGBoost | F1 `0.666`, ROC AUC `0.832` |
| OULAD day-35 landmark | XGBoost | F1 `0.403`, ROC AUC `0.717`, AP `0.356` |
| OULAD 2014J holdout | XGBoost | F1 `0.301`, ROC AUC `0.678`, AP `0.229` |

In the fixed day-75 cohort, ROC AUC increased from `0.673` at day 35 to
`0.700` at day 60 and `0.718` at day 75. In the changing operational cohorts,
ROC AUC stayed near `0.717` while average precision declined as early
withdrawals left the risk set. The old Milestone 4-6 scores remain in their
historical folders, but they are preliminary classification results and should
not be presented as prospective early-warning estimates.

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
    final_report/
```

Each milestone folder has a short README, the submitted PDF, and the code or
results used at that stage. Word drafts, report-building scripts, meeting
transcripts, rendered previews, and superseded slide decks stay local.

The course report is available at
[`capstone/work/final_report/Final_Capstone_Report.pdf`](capstone/work/final_report/Final_Capstone_Report.pdf).
A separate working manuscript is available at
[`capstone/work/final_report/Journal_Style_Manuscript.pdf`](capstone/work/final_report/Journal_Style_Manuscript.pdf).

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
python3 capstone/work/final_report/final_analysis.py
```

The main packages are pandas, NumPy, Matplotlib, scikit-learn, XGBoost, and
SHAP. The scripts use fixed random seeds so the saved tables can be reproduced.

## Data Note

The UCI source file and the OULAD source archive are under
`capstone/work/milestone3/data/raw/`. The extracted OULAD `studentVle.csv` is
not tracked because it is about 433 MB. The scripts can read that table from
`oulad.zip` instead.
