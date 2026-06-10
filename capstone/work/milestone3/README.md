# Milestone 3

This folder contains the third milestone of the capstone project.

## Milestone 3 Goal

Milestone 3 focused on:

- collecting the final project datasets
- performing initial exploratory data analysis
- evaluating data quality and modeling readiness

This milestone was the bridge between project design and actual model building.

## What We Did

### Step 1: Finalized the project datasets

- Collected the `UCI Predict Students' Dropout and Academic Success` dataset.
- Collected the `Open University Learning Analytics Dataset (OULAD)`.
- Stored the raw files locally under `data/raw/`.

### Step 2: Verified the structure and quality of both datasets

- Confirmed row counts, columns, target variables, missing values, and duplicates.
- For `UCI`:
  - confirmed `4,424` records and `37` columns
  - confirmed no missing values
  - confirmed no duplicate rows
- For `OULAD`:
  - confirmed the multi-table structure
  - reviewed `studentInfo`, `studentRegistration`, `studentAssessment`, `studentVle`, `assessments`, `courses`, and `vle`

### Step 3: Performed initial EDA on UCI

- Reviewed target distribution across:
  - `Graduate`
  - `Dropout`
  - `Enrolled`
- Compared key variables across outcome groups.
- Ran selected correlation analysis.
- Confirmed that approved curricular units and semester grades were strongly associated with student outcomes.

### Step 4: Performed initial EDA on OULAD

- Reviewed `final_result` distribution across:
  - `Pass`
  - `Withdrawn`
  - `Fail`
  - `Distinction`
- Compared key attributes such as studied credits, assessment participation, scores, and VLE activity.
- Confirmed that withdrawn students had much lower assessment participation and much lower VLE engagement than successful students.

### Step 5: Tested whether OULAD could realistically support later modeling

- Verified the multi-table join structure using:
  - `code_module`
  - `code_presentation`
  - `id_student`
- Confirmed that `studentVle` was large but still manageable through chunked aggregation.
- Confirmed that OULAD was suitable for later student-level feature construction.

### Step 6: Identified the main preprocessing risks

- In `UCI`:
  - many coded fields should be handled categorically or ordinally
  - later-semester variables may create early-warning leakage concerns
- In `OULAD`:
  - multi-table integration is required
  - behavior logs must be aggregated before modeling
  - `date_unregistration` and late-course behavior can leak the outcome

## Key Findings

- `UCI` was already clean enough to move directly into baseline modeling.
- `OULAD` was much richer but not modeling-ready in raw form.
- Withdrawn students in OULAD showed much weaker assessment participation and much lower VLE engagement than successful students.
- The biggest Milestone 3 value was not model performance yet; it was proving that the project had usable data and a clear preprocessing path.

## What This Milestone Produced

- The Milestone 3 report
- The Milestone 3 structure document
- EDA code and report-builder code
- Local figures and summary outputs under `data/`

## Main Files

- `Activity 3-1.pdf`
- `Milestone3.pdf`
- `Milestone3_deliverable.docx`
- `Milestone3_deliverable_prof_style.docx`
- `Milestone3_structure.docx`
- `code/milestone3_eda_pipeline.py`
- `code/milestone3_report_builder.py`

## Data Outputs

- `data/analysis/milestone3_eda_summary.json`
- `data/figures/`
- `data/raw/`

## Why Milestone 3 Matters

Milestone 3 established that:

- `UCI` was immediately ready for baseline modeling
- `OULAD` was valuable but needed aggregation and leakage-aware preprocessing first

That conclusion directly led into Milestone 4, where we actually built the preprocessing pipeline and ran the first baseline models.

## What Milestone 3 Did Not Do Yet

- It did not run the final baseline models.
- It did not complete the student-level early-warning feature pipeline for OULAD.
- It did not yet test cross-dataset generalization.

Those tasks became the focus of Milestone 4.
