# AI-Based Student Dropout Early Warning System

## Overview

This package contains three related analyses:

1. A portable 16-feature dynamic candidate that uses one fixed numeric schema
   at enrolment and at later course snapshots.
2. The earlier portable six-feature enrolment baseline.
3. A time-specific 51-feature OULAD benchmark using assessment and learning-
   platform activity through days 35, 60, and 75.

The dynamic candidate replaces assignment numbers and platform-specific click
types with semantic rates such as assessment completion, normalized average
score, active-day rate, and activity recency. Optional feature groups may be
blank. Training masks optional groups so the input schema remains usable when
another institution lacks an assessment or LMS feed.

UCI does not contain dated assessment or LMS events, so it contributes
enrolment snapshots while OULAD contributes enrolment and day-35/60/75
snapshots. The same pooled model object scores all snapshots. This is a
portable dynamic prototype, not completed cross-institution validation of the
dynamic fields. A third institution with dated events is still required for
that claim.

## Project Structure

```text
Milestone8_Code_Data_Package/
|-- code/
|   |-- cross_school_dynamic_model.py
|   |-- map_new_school_snapshots.py
|   |-- score_new_school_dynamic.py
|   |-- cross_school_unified_model.py
|   |-- score_new_school.py
|   |-- student_dropout_analysis.py
|   |-- feature_preparation.py
|   `-- check_outputs.py
|-- data/
|   `-- raw/
|       |-- uci_student_dropout.csv
|       `-- oulad/
|           |-- oulad.zip
|           |-- assessments.csv
|           |-- courses.csv
|           |-- studentAssessment.csv
|           |-- studentInfo.csv
|           |-- studentRegistration.csv
|           `-- vle.csv
|-- examples/
|   |-- new_school_dynamic_features.csv
|   |-- new_school_mapping.json
|   |-- new_school_raw_snapshots.csv
|   `-- new_school_enrolment_features.csv
|-- models/
|   |-- unified_dynamic_xgboost.joblib
|   |-- unified_dynamic_feature_contract.json
|   |-- unified_enrolment_xgboost.joblib
|   `-- unified_enrolment_feature_contract.json
|-- outputs/
|   |-- figures/
|   `-- results/
|-- README.md
`-- requirements.txt
```

`cross_school_dynamic_model.py` constructs the fixed 16-feature snapshots,
runs source-only and pooled experiments, and saves the frozen dynamic model.
`map_new_school_snapshots.py` converts institution-specific column names and
scales using a JSON configuration. `score_new_school_dynamic.py` then scores
the unlabeled standardized snapshot CSV and reports feature coverage. The
earlier enrolment baseline remains in
`cross_school_unified_model.py`, and `student_dropout_analysis.py` contains the
larger OULAD-specific upper benchmark.

## Portable Dynamic Input Contract

All model inputs are numeric values on documented 0-1 scales. No raw course
code, region, assignment number, or platform-specific activity type enters the
model.

Required:

- `age_scaled`
- `prior_education_level`
- `study_load`
- `course_progress_ratio`

Optional background fields:

- `prior_academic_score`
- `previous_attempts`
- `male`
- `declared_support_need`
- `financial_stability`

Optional assessment fields:

- `assessment_completion_rate`
- `assessment_average_score`
- `assessment_score_available`
- `late_submission_rate`

Optional platform-neutral activity fields:

- `active_day_rate`
- `days_since_last_activity_scaled`
- `recent_activity_rate`

Optional fields may be absent or blank. The model uses training medians, and
structured feature-group masking during training reduces dependence on a feed
being available. Missing required fields are rejected. Definitions and source
mappings are stored in `models/unified_dynamic_feature_contract.json`.

The output is an uncalibrated ranking score. Use within-course or within-cohort
percentiles for review; do not present the raw score as an individual dropout
probability.

Main pooled dynamic-candidate results:

| Test snapshot | ROC-AUC | Average precision | F1 |
| --- | ---: | ---: | ---: |
| UCI enrolment | 0.771 | 0.635 | 0.599 |
| OULAD enrolment | 0.595 | 0.310 | 0.384 |
| OULAD day 35 | 0.677 | 0.311 | 0.367 |
| OULAD day 60 | 0.694 | 0.292 | 0.359 |
| OULAD day 75 | 0.689 | 0.270 | 0.322 |

F1 thresholds were selected using validation data separately at each snapshot
day. Operational scoring uses percentiles because a threshold learned from the
two source datasets is not automatically valid at a new institution. The
portable candidate intentionally performs below the OULAD-specific 51-feature
model at day 35 (ROC-AUC 0.717, F1 0.403); that gap is the measured cost of
removing platform- and course-specific predictors. The OULAD-only portable
model also had slightly higher ROC-AUC than the pooled model at days 35, 60,
and 75. Paired student-cluster permutation tests found the differences in
ROC-AUC significant (p = 0.005, 0.001, and 0.005), although the decreases were
small at 0.009, 0.013, and 0.009. The corresponding average-precision
differences were not significant at 0.05.

The saved dynamic model was trained with equal total weight for each
institution and outcome class. Students with several module snapshots were
also down-weighted so they did not dominate the training objective. SHAP
summaries are reported separately by dataset and snapshot because the same
model can rely on its inputs differently as course information accumulates.

## Portable Enrolment Input Contract

The pooled model uses six values on a 0-1 scale.

Required:

- `age_scaled`
- `prior_preparation`
- `study_load`

Optional:

- `male`
- `declared_support_need`
- `financial_stability`

Optional columns may be absent or blank. The fitted training median is used in
that case. A new school must document how each supplied value maps to the
definitions in `models/unified_enrolment_feature_contract.json`. Renaming an
unrelated local variable to a shared name is not valid feature harmonization.

The model output is an uncalibrated ranking score. It should be presented as a
within-course or within-cohort risk percentile, not as an individual withdrawal
probability. Demographic and support fields should be included only when local
governance permits their use.

## What the Portability Experiments Show

The same pooled XGBoost object is evaluated on held-out UCI and OULAD records.
Pooled evaluation shows whether one estimator can fit both known data
contexts. Zero-shot evaluation is the closer test of portability because the
target dataset is absent from training.

Main enrolment-feature results:

| Experiment | ROC-AUC | Average precision |
| --- | ---: | ---: |
| UCI local model to UCI | 0.777 | 0.644 |
| OULAD local model to OULAD | 0.635 | 0.418 |
| UCI zero-shot to OULAD | 0.481 | 0.298 |
| OULAD zero-shot to UCI | 0.603 | 0.407 |
| Pooled model to UCI | 0.777 | 0.635 |
| Pooled model to OULAD | 0.624 | 0.409 |

These results do not establish external institutional validity. A third school
must remain completely absent from training and model development, then be
used as the final external test. Local adaptation experiments are reported
separately because using labeled target-school records is not zero-shot use.

## Data Sources

- UCI Predict Students' Dropout and Academic Success:
  <https://archive.ics.uci.edu/dataset/697>
- Open University Learning Analytics Dataset:
  <https://analyse.kmi.open.ac.uk/open_dataset>

The original OULAD archive is included as `data/raw/oulad/oulad.zip`. The
smaller source tables are also included separately. The large
`studentVle.csv` table is read directly from the ZIP by the temporal analysis.
The portable enrolment model reads only `studentInfo.csv` and does not load the
large event table.

## Dependencies

- Python 3.11 or later
- NumPy
- pandas
- scikit-learn
- SciPy
- Matplotlib
- joblib
- XGBoost
- SHAP

Exact package versions are listed in `requirements.txt`.

## Train and Validate

Windows:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py code\cross_school_dynamic_model.py
py code\cross_school_unified_model.py
py code\student_dropout_analysis.py
py code\check_outputs.py
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 code/cross_school_dynamic_model.py
python3 code/cross_school_unified_model.py
python3 code/student_dropout_analysis.py
python3 code/check_outputs.py
```

The random seed is fixed at 42. The dynamic and OULAD-specific analyses process
more than ten million interaction records. The enrolment-only baseline runs
without that table.

## Score a New School

Start with `examples/new_school_dynamic_features.csv`. The scoring file does
not require `target`. Registration-time rows set `course_progress_ratio` to 0
and may leave all assessment and activity values blank.

If local column names and scales differ, first edit
`examples/new_school_mapping.json`, then map the raw snapshots:

```bat
py code\map_new_school_snapshots.py ^
  examples\new_school_raw_snapshots.csv ^
  examples\new_school_mapping.json ^
  outputs\results\new_school_mapped_snapshots.csv
```

The mapping configuration documents local GPA maxima, heavy study load,
course length, inactivity cap, education levels, and binary-value mappings.
It changes columns and units only; it does not fit or modify the model.
Required values, direct 0-1 rates, binary fields, and course length are checked
before a mapped file is accepted. This is intended to catch an incorrect local
unit or mapping instead of silently treating it as a valid model input.

Windows:

```bat
py code\score_new_school_dynamic.py ^
  examples\new_school_dynamic_features.csv ^
  outputs\results\new_school_dynamic_scores.csv ^
  --group-column course_id,snapshot_day ^
  --review-share 0.15 ^
  --coverage-output outputs\results\new_school_feature_coverage.csv
```

macOS or Linux:

```bash
python3 code/score_new_school_dynamic.py \
  examples/new_school_dynamic_features.csv \
  outputs/results/new_school_dynamic_scores.csv \
  --group-column course_id,snapshot_day \
  --review-share 0.15 \
  --coverage-output outputs/results/new_school_feature_coverage.csv
```

The output contains `risk_score`, `risk_percentile`, and `review_flag`. The
default flag identifies approximately the highest-ranked 15% within each
specified course and snapshot day. A real institution should first run the
frozen model on an
untouched historical cohort and report ROC-AUC, average precision, subgroup
results, and score stability before using the list operationally.

## Expected Temporal-Analysis Results

- UCI day-one XGBoost ROC-AUC: approximately 0.832.
- OULAD day-35 XGBoost ROC-AUC: approximately 0.717.
- OULAD day-35 XGBoost average precision: approximately 0.356.
- OULAD 2014J XGBoost ROC-AUC: approximately 0.678.
- Student overlap between train, validation, and test: zero.

Small changes in library versions can cause minor numerical differences.

## Contact

Xinyu Wang  
GitHub: <https://github.com/bwaj287>
