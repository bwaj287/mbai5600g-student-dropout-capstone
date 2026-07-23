from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def print_step(number, title):
    print("\n" + "=" * 70)
    print(f"Step {number} - {title}")
    print("=" * 70)


print_step(1, "Load the Three Time-Window Results")

metrics = pd.read_csv(RESULTS / "temporal_model_metrics.csv")
metrics.info()
print(metrics.head())


print_step(2, "Compare Test Performance")

performance_columns = [
    "window_days",
    "fixed_threshold",
    "validation_selected_threshold",
    "fixed_precision",
    "fixed_recall",
    "fixed_f1",
    "roc_auc",
    "average_precision",
]
print(metrics[performance_columns].round(4).to_string(index=False))


print_step(3, "Check the 95% Confidence Intervals")

confidence_columns = [
    "window_days",
    "fixed_f1_ci_95_lower",
    "fixed_f1_ci_95_upper",
    "roc_auc_ci_95_lower",
    "roc_auc_ci_95_upper",
]
print(metrics[confidence_columns].round(4).to_string(index=False))


print_step(4, "Review Paired Window Differences")

differences = pd.read_csv(RESULTS / "paired_window_differences.csv")
print(differences.round(4).to_string(index=False))


print_step(5, "Compare SHAP Feature Families")

shap_families = pd.read_csv(RESULTS / "shap_family_by_window.csv")
family_table = shap_families.pivot_table(
    index="feature_family",
    columns="window_days",
    values="share_of_total_shap",
    fill_value=0,
)
print((family_table * 100).round(1).to_string())


print_step(6, "Review Top SHAP Features at Each Window")

for day in [35, 60, 75]:
    shap_results = pd.read_csv(
        RESULTS / f"day_{day}_shap_summary.csv"
    )
    print(f"\nTop five features through day {day}")
    print(
        shap_results[
            ["feature", "mean_abs_shap", "feature_family"]
        ]
        .head(5)
        .round(4)
        .to_string(index=False)
    )


print_step(7, "Main Interpretation")

print(
    "The fixed-threshold F1 score increases from day 35 to day 75. "
    "The paired bootstrap intervals are above zero, so the improvement "
    "is not only a small change in one test score."
)
print(
    "SHAP also changes over time. Early academic progress becomes more "
    "important as more assessments are available, while program setup "
    "and prior preparation become less important."
)
