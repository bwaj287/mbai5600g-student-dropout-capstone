# Milestone 5: Model Development

Milestone 5 tuned the strongest baseline models and added model interpretation.
The search was intentionally small enough to run locally and to keep the
comparison understandable.

## Experiments

- Tuned XGBoost for the UCI multiclass task.
- Tuned Logistic Regression and XGBoost for UCI early warning.
- Tuned XGBoost for OULAD early warning.
- Selected binary decision thresholds on the validation set.
- Compared tuned test results with the untouched Milestone 4 test results.
- Calculated SHAP values and grouped features into common families.

| Task and model | Baseline | Tuned test result | Change |
| --- | ---: | ---: | ---: |
| UCI multiclass XGBoost, macro F1 | `0.704` | `0.696` | `-0.008` |
| UCI binary Logistic Regression, F1 | `0.783` | `0.793` | `+0.010` |
| UCI binary XGBoost, F1 | `0.783` | `0.767` | `-0.016` |
| OULAD binary XGBoost, F1 | `0.729` | `0.738` | `+0.009` |

Tuning did not improve every model. The UCI Logistic Regression and OULAD
XGBoost became the final early-warning candidates. The decreases are reported
because they show that validation-set tuning does not guarantee a better test
result.

## Interpretation

UCI early warning depended mainly on academic progress, financial status, and
demographics. OULAD also depended on academic progress, but online engagement
and course setup contributed more. This comparison suggested that warning
signals depend on what the institution records.

## Files

- `milestone5_model_development.py`: tuning, threshold selection, and SHAP
- `analyze_model_explanations.py`: feature-family comparison
- `modeling_results_walkthrough.py`: short review of the saved results
- `results/tuned_model_comparison.csv`: baseline and tuned results
- `Milestone5.pdf`: milestone report

## Run

```bash
python3 capstone/work/milestone5/milestone5_model_development.py
python3 capstone/work/milestone5/analyze_model_explanations.py
python3 capstone/work/milestone5/modeling_results_walkthrough.py
```
