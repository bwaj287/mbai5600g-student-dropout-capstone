## Milestone 3 Code

This folder contains the scripts used to prepare the Milestone 3 report.

### Files

- `milestone3_eda_pipeline.py`
  - Loads the UCI and OULAD datasets
  - Runs the initial EDA used in the report
  - Saves summary results to `capstone/work/milestone3/data/analysis/milestone3_eda_summary.json`
  - Saves charts to `capstone/work/milestone3/data/figures/`

- `milestone3_report_builder.py`
  - Reads the EDA summary JSON and figures
  - Builds the Word report for Milestone 3
  - Outputs local Word drafts that are kept out of the public remote

### Recommended Run Order

1. Run `milestone3_eda_pipeline.py`
2. Run `milestone3_report_builder.py`

### Notes

- The scripts now resolve project paths relative to their own location.
- The datasets are stored under `capstone/work/milestone3/data/raw/`.
- In the OULAD dataset, `vle` means `Virtual Learning Environment`, and `studentVle` is the student-level Virtual Learning Environment activity table from the original dataset.
- The analysis artifacts under `capstone/work/milestone3/data/` are kept alongside the Milestone 3 workspace.
