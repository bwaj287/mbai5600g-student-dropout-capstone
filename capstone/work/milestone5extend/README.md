# Milestone 5 Extension: Warning Time

This follow-up experiment addresses feedback from the Milestone 5 review. It
tests the selected OULAD XGBoost setup with activity available through days 35,
60, and 75.

## Fair Comparison

All three windows use the same student records, target, data split, model
settings, and fixed test threshold of `0.60`. Only the available assessment and
Virtual Learning Environment history changes.

| OULAD window | Test F1 |
| --- | ---: |
| Day 35 | `0.682` |
| Day 60 | `0.716` |
| Day 75 | `0.738` |

The result shows a practical trade-off. Waiting for more activity improved
performance, while an earlier warning would give staff more time to contact a
student. Bootstrap confidence intervals and paired prediction comparisons are
included because the score differences are not large enough to interpret from
point estimates alone.

## Files

- `temporal_early_warning_analysis.py`: builds and evaluates the three windows
- `temporal_results_walkthrough.py`: prints the saved results in a short format
- `results/`: metrics, confidence intervals, threshold checks, and SHAP tables
- `figures/`: performance and explanation charts
- `Milestone5_Temporal_Extension_Results.pdf`: results summary

## Run

```bash
python3 capstone/work/milestone5extend/temporal_early_warning_analysis.py
python3 capstone/work/milestone5extend/temporal_results_walkthrough.py
```
