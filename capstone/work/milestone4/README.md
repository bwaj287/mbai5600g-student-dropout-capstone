# Milestone 4: Preparation And Baselines

Milestone 4 converted both datasets into model-ready tables and established the
baseline results used in Milestone 5.

## Preparation

- Created a three-class UCI task for comparison with the base paper.
- Created a binary UCI early-warning task and removed second-semester fields.
- Joined the OULAD tables by module, presentation, and student.
- Limited OULAD assessment and Virtual Learning Environment activity to the
  first 75 days.
- Removed `date_unregistration` and other outcome leakage.
- Grouped columns into shared feature families for later interpretation.

## Baseline Models

The same five model families were evaluated for each task: Logistic Regression,
Decision Tree, Random Forest, Gradient Boosting, and XGBoost. We used
stratified train, validation, and test sets. The selected model also received
five-fold cross-validation.

| Task | Selected model | Test result |
| --- | --- | --- |
| UCI multiclass | XGBoost | Accuracy `0.768`, macro F1 `0.704` |
| UCI binary early warning | Logistic Regression | F1 `0.783`, ROC AUC `0.910` |
| OULAD binary early warning | XGBoost | F1 `0.729`, ROC AUC `0.885` |

The UCI multiclass result was below the `0.83` accuracy reported by the base
paper. We treated this as a reproduction benchmark, not an exact replication,
because the paper did not provide every implementation detail.

## Files

- `run_milestone4_prep.py`: creates the model-ready tables
- `run_milestone4_baselines.py`: trains and compares the baseline models
- `results/baseline_comparison.csv`: compact result table
- `data/shared_feature_schema.json`: feature-family mapping
- `Milestone4.pdf`: milestone report

## Run

```bash
python3 capstone/work/milestone4/run_milestone4_prep.py
python3 capstone/work/milestone4/run_milestone4_baselines.py
```
