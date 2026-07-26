# Final Report and Journal Manuscript

This folder contains two integrated versions of the project.

## Course Final Report

- `Final_Capstone_Report.pdf` is the course-facing report.
- `Final_Capstone_Report.docx` is the local editable copy and is not tracked.

This version includes an executive summary, table of contents, business use,
deployment guidance, risk review, reproducibility notes, and course-deliverable
traceability.

## Journal-Style Manuscript

- `Journal_Style_Manuscript.pdf` is a compact manuscript-style version.
- `Journal_Style_Manuscript.docx` is the local editable copy and is not tracked.

This version uses a structured abstract, manuscript sections, data and code
availability statements, an ethics statement, competing-interest disclosure,
and author-contribution wording. It is a working manuscript, not a
submission-ready claim of major novelty. A target journal template, a
capacity-linked fairness audit, and a survival-model comparison are still
recommended before submission.

## Final Analysis

`final_analysis.py` replaces the preliminary milestone evaluation with:

1. OULAD landmark cohorts containing only students still enrolled at days 35,
   60, and 75.
2. Future withdrawal after each cutoff as the outcome.
3. Mutually exclusive student-level training, validation, and test groups.
4. Model settings selected on the day-35 validation set and fixed across later
   windows.
5. Validation-only probability calibration and threshold selection.
6. Student-cluster bootstrap confidence intervals and paired temporal tests.
7. A controlled same-record temporal comparison.
8. A later 2014J presentation holdout.
9. A UCI enrolment-time benchmark that removes semester-result variables.

The generated tables are in `journal_results/`, and figures are in
`journal_figures/`.

## Interpretation Note

The original milestone UCI and OULAD scores were preliminary classification
results. They should not be presented as prospective early-warning estimates.
The final analysis reports lower but more defensible performance after
correcting student overlap and outcome-timing leakage.
