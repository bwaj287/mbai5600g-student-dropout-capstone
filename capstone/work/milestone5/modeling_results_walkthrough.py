from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


print_section("Milestone 5 - Advanced Modeling and Optimization")

print(
    "This walkthrough summarizes the Milestone 5 modeling outputs in the same "
    "step-by-step style used in the course sample notebooks."
)

print_section("Step 1 - Load Tuned Model Comparison Results")

comparison = pd.read_csv(RESULTS / "tuned_model_comparison.csv")
print(comparison.info())
print(comparison.head())

print_section("Step 2 - Compare Baseline and Tuned Model Performance")

summary_columns = [
    "task",
    "model",
    "primary_metric",
    "baseline_model",
    "baseline_test_primary_metric",
    "test_primary_metric",
    "test_primary_metric_change_vs_baseline",
    "selected_threshold",
]
print(comparison[summary_columns].to_string(index=False))

print_section("Step 3 - Identify Best Candidate Models")

for task_name, task_frame in comparison.groupby("task"):
    best_row = task_frame.sort_values("test_primary_metric", ascending=False).iloc[0]
    print(f"\nTask: {task_name}")
    print(f"Best tuned model: {best_row['model']}")
    print(f"Primary metric: {best_row['primary_metric']}")
    print(f"Test primary metric: {best_row['test_primary_metric']:.4f}")
    print(f"Change vs baseline: {best_row['test_primary_metric_change_vs_baseline']:.4f}")
    if not pd.isna(best_row["selected_threshold"]):
        print(f"Selected threshold: {best_row['selected_threshold']:.2f}")

print_section("Step 4 - Review SHAP Feature Importance Outputs")

shap_files = sorted(RESULTS.glob("*_shap_summary.csv"))
for shap_file in shap_files:
    print(f"\nTop SHAP features from {shap_file.name}")
    shap_summary = pd.read_csv(shap_file)
    print(shap_summary.head(10).to_string(index=False))

print_section("Step 5 - Review Feature-Family Explanation Shift")

family_path = RESULTS / "shap_feature_family_comparison.csv"
if family_path.exists():
    family_summary = pd.read_csv(family_path)
    print(family_summary.to_string(index=False))
else:
    print("Feature-family comparison has not been generated yet.")

print_section("Step 6 - Practical Interpretation")

print(
    "The tuning results show that model complexity does not automatically improve "
    "performance. Logistic Regression remains the strongest UCI early-warning "
    "candidate, while XGBoost is more useful for the larger and more complex OULAD "
    "early-warning dataset."
)

print(
    "The SHAP outputs also show that UCI and OULAD rely on different predictive "
    "signals. UCI is driven mainly by academic progress and financial indicators, "
    "while OULAD is driven mainly by early assessment participation, scores, and "
    "learning-platform engagement."
)
