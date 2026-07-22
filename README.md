# MBAI 5600G Capstone Project

This repository contains the working documents for an MBAI 5600G Applied Integrative Analytics Capstone Project focused on student dropout prediction in higher education.

The current project direction is:

`AI-Based Student Dropout Early Warning System for Higher Education: Reproduction and Cross-Dataset Generalization on UCI and OULAD`

## Project Overview

The capstone studies whether explainable machine learning workflows for student risk prediction remain effective when moved across different educational datasets.

The project has two linked goals:

- Reproduce the explainable machine learning workflow described by Islam et al. (2025) on the UCI Predict Students' Dropout and Academic Success dataset.
- Extend that work by evaluating robustness, transferability, and explanation stability using the Open University Learning Analytics Dataset (OULAD).

## Current Status

The project has completed Milestones 1-4 and now includes a Milestone 5 advanced-modeling draft.

- `Milestone 1`: finalized the capstone topic and business problem.
- `Milestone 2`: completed the literature review and confirmed the two-dataset project design.
- `Milestone 3`: collected the UCI and OULAD datasets, performed initial EDA, and identified the main preprocessing and leakage risks.
- `Milestone 4`: built preprocessing pipelines and ran baseline models on both datasets.
- `Milestone 5`: tuned selected models, compared them with baselines, and prepared SHAP-based explanation outputs.

The current working direction is no longer just "build one dropout model." It is now:

- reproduce a benchmark explainable ML workflow on UCI
- build an early-warning pipeline on OULAD
- compare whether predictive signals and explanations remain stable across datasets

## Repository Structure

```text
capstone/
  material/    code samples and a local reference-material index
  other/       private administrative documents kept out of the public repo
  work/
    milestone1/
    milestone2/
    milestone3/
      data/
      code/
    milestone4/
    milestone5/
```

## Milestone Snapshot

| Milestone | Main focus | Main outcome | Folder |
| --- | --- | --- | --- |
| 1 | Project proposal | Defined the business problem and initial ML direction | `capstone/work/milestone1/` |
| 2 | Literature review and data source identification | Reframed the project as reproduction plus cross-dataset extension | `capstone/work/milestone2/` |
| 3 | Data collection and initial EDA | Confirmed dataset quality, EDA findings, and preprocessing risks | `capstone/work/milestone3/` |
| 4 | Preprocessing and baseline modeling | Built model-ready datasets and ran baseline models | `capstone/work/milestone4/` |
| 5 | Advanced modeling and optimization | Tuned selected models and generated model explanation outputs | `capstone/work/milestone5/` |

Each milestone folder now includes its own README so the work can be understood stage by stage without reading the whole repository at once.

## Current Modeling Position

Right now the project is in the advanced modeling stage.

- `UCI` has been prepared both as:
  - a multiclass benchmark reproduction task
  - an early-warning binary attrition task
- `OULAD` has been converted from multi-table raw data into a student-level early-warning modeling table
- the baseline workflow uses stratified `60/20/20` train-validation-test splits with `5-fold` cross-validation on the selected model
- the Milestone 5 workflow adds focused hyperparameter tuning, threshold tuning, and SHAP-based explanation outputs

## Current Baseline Results

These are the main baseline results from `capstone/work/milestone4/`.

| Task | Best model | Main metrics |
| --- | --- | --- |
| UCI multiclass reproduction | XGBoost | accuracy `0.768`, macro F1 `0.704` |
| UCI binary early warning | Logistic regression | accuracy `0.858`, F1 `0.783`, ROC AUC `0.910` |
| OULAD binary early warning | XGBoost | accuracy `0.843`, F1 `0.729`, ROC AUC `0.885` |

## Current Tuned Results

These are the main tuned results from `capstone/work/milestone5/`.

| Task | Tuned model | Main result |
| --- | --- | --- |
| UCI multiclass reproduction | XGBoost | macro F1 `0.696`, slightly below the Milestone 4 baseline |
| UCI binary early warning | Logistic regression | F1 `0.793`, about `+0.010` above the baseline |
| OULAD binary early warning | XGBoost | F1 `0.738`, about `+0.009` above the baseline |

## Core Terms

### VLE

`VLE` means `Virtual Learning Environment`.

- In OULAD, this is the online learning platform behavior data.
- It gives us early engagement signals such as clicks, activity, and interaction patterns.

### Shared Feature Schema

`shared_feature_schema.json` groups variables into common concept families.

- It does not mean UCI and OULAD have identical columns.
- It means we align them at the concept level so later comparisons are fair.

### F1

`F1` balances precision and recall.

- It is useful when we care about both catching at-risk students and avoiding too many false alarms.

### ROC AUC

`ROC AUC` measures how well the model separates higher-risk from lower-risk students overall.

- It is more about overall ranking quality than one fixed cutoff.

## Where To Read Next

If you want the clearest step-by-step view of the project, read the files in this order:

- `capstone/work/milestone1/README.md`
- `capstone/work/milestone2/README.md`
- `capstone/work/milestone3/README.md`
- `capstone/work/milestone4/README.md`
- `capstone/work/milestone5/README.md`

If you want the latest technical outputs first, start with:

- `capstone/work/milestone5/results/tuned_model_comparison.csv`
- `capstone/work/milestone5/results/shap_feature_family_comparison.csv`
- `capstone/work/milestone5/Milestone5.pdf`

## Included Content

This public repository currently includes:

- milestone requirement PDFs used for course planning
- milestone draft documents and revision guides
- milestone structure documents for follow-up work
- selected sample notebooks and sample datasets used to study course coding style
- project-level documentation for organizing the capstone

## Excluded Content

Some local files were intentionally not published in this public repository:

- personal and administrative documents such as payment files, forms, and transcript-related records
- local course reading PDFs and ZIP files stored for private study and reference

These exclusions were made for privacy and copyright reasons.

## Coding Style Reference

The repository also includes a small `capstone/material/code sample/` folder containing example notebooks and sample datasets used to study the professor's preferred instructional coding style. These samples suggest a style that is:

- step-by-step and notebook-friendly
- explicit in imports, transformations, and print outputs
- light on abstraction and heavy on clarity
- organized around explanation, preprocessing, visualization, and interpretation
- suitable for academic demonstrations rather than production engineering

## Data Plan

The intended analytical workflow uses:

- `UCI Predict Students' Dropout and Academic Success` as the reproduction and benchmark dataset
- `OULAD` as the primary evaluation dataset for richer behavioural and temporal analysis

The modeling direction emphasizes:

- reproduction of the base paper
- cross-dataset generalization
- explanation stability
- readiness for baseline and advanced modeling in later milestones

## Notes

This repository remains document-focused overall, but the local working tree also includes local milestone data and generated analysis artifacts used for Milestones 3, 4, and 5.

Milestone Word drafts are kept locally and are not intended for the public remote. The repository version should use PDF milestone reports.

For `OULAD`, the repository tracks the reusable source archive `capstone/work/milestone3/data/raw/oulad/oulad.zip`. The extracted `studentVle.csv` table is intentionally left out of git because it is roughly `433MB`, above GitHub's normal single-file limit, and the milestone scripts now fall back to the archive when that extracted file is absent.

## License

No license has been added yet. If this repository is later expanded to include code or reusable assets, a project license can be added at that stage.
