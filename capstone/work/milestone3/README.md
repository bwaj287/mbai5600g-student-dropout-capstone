# Milestone 3: Data And EDA

Milestone 3 collected the two datasets and checked whether they could support
the proposed analysis.

## Data Checked

- UCI: `4,424` student records and `37` columns
- OULAD: linked tables for student information, registration, assessments,
  courses, Virtual Learning Environment activity, and final results

The UCI file had no missing values or duplicate rows. OULAD required several
table joins and chunked processing of `studentVle.csv`.

## Main EDA Findings

- UCI approved units and semester grades were strongly related to the final
  student outcome.
- OULAD withdrawn students generally submitted fewer assessments and had less
  online activity.
- UCI second-semester fields would be too late for an early warning.
- OULAD `date_unregistration` and activity after the warning date would leak
  information about the outcome.

## Files

- `Milestone3.pdf`: milestone report
- `code/milestone3_eda_pipeline.py`: EDA script
- `data/analysis/`: saved summary tables
- `data/figures/`: EDA charts
- `data/raw/`: source data

The extracted OULAD `studentVle.csv` is not tracked because of its size. The
EDA script reads the same file directly from `oulad.zip` when needed.
