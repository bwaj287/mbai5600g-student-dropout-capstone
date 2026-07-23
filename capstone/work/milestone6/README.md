# Milestone 6: Validation And Impact

Milestone 6 tested the two final early-warning candidates beyond one test score.
No new model family was introduced.

## Work Completed

- Repeated the final held-out test evaluation.
- Ran five additional stratified train-test splits.
- Simulated 10% missing feature values and 5% numeric noise.
- Reviewed false positives and false negatives.
- Compared performance across selected student groups.
- Converted the results into expected alerts per 1,000 students.

| Dataset | Final model | F1 | Recall | ROC AUC |
| --- | --- | ---: | ---: | ---: |
| UCI | Logistic Regression | `0.793` | `0.820` | `0.911` |
| OULAD | XGBoost | `0.738` | `0.719` | `0.885` |

Small numeric noise had little effect. Missing values caused the clearest
performance decline, especially for OULAD. The impact calculation estimated
about 344 UCI alerts and 295 OULAD alerts per 1,000 records. These are workload
estimates, not proof that an intervention will prevent dropout.

The recommendation is a monitored pilot where advisors use the score as one
source of information. It should not make automatic academic decisions.

## Files

- `milestone6_validation_impact.py`: validation and impact analysis
- `results/`: validation, robustness, error, segment, and impact tables
- `figures/`: report figures
- `Milestone6.pdf`: milestone report

## Run

```bash
python3 capstone/work/milestone6/milestone6_validation_impact.py
```
