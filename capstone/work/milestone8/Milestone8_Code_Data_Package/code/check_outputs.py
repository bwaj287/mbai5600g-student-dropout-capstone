import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results"
MODELS = ROOT / "models"
EXAMPLES = ROOT / "examples"


def main():
    metrics = pd.read_csv(RESULTS / "journal_model_metrics.csv")
    overlap = pd.read_csv(RESULTS / "split_overlap_check.csv")
    paired = pd.read_csv(RESULTS / "journal_paired_differences.csv")
    unified = pd.read_csv(RESULTS / "unified_model_metrics.csv")
    unified_tests = pd.read_csv(
        RESULTS / "unified_model_significance.csv"
    )
    shared_schema = pd.read_csv(
        RESULTS / "unified_shared_feature_schema.csv"
    )
    dynamic = pd.read_csv(
        RESULTS / "dynamic_unified_model_metrics.csv"
    )
    dynamic_schema = pd.read_csv(
        RESULTS / "dynamic_unified_feature_schema.csv"
    )
    dynamic_overlap = pd.read_csv(
        RESULTS / "dynamic_unified_split_overlap.csv"
    )
    dynamic_tests = pd.read_csv(
        RESULTS / "dynamic_unified_significance.csv"
    )
    dynamic_shap = pd.read_csv(
        RESULTS / "dynamic_unified_shap.csv"
    )

    required_metric_columns = {
        "cohort",
        "cutoff_day",
        "model",
        "f1",
        "roc_auc",
        "average_precision",
        "brier_score",
    }
    missing = required_metric_columns - set(metrics.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")

    overlap_columns = [
        column
        for column in overlap.columns
        if column.endswith("_overlap")
    ]
    if overlap[overlap_columns].to_numpy().max() != 0:
        raise ValueError("Student overlap was found across data splits.")

    day_35 = metrics.loc[
        (metrics["cohort"] == "dynamic_landmark")
        & (metrics["cutoff_day"] == 35)
        & (metrics["model"] == "XGBoost")
    ]
    if len(day_35) != 1:
        raise ValueError("The day-35 XGBoost result is missing.")

    if len(paired) != 6:
        raise ValueError("The paired temporal result table is incomplete.")

    unified_xgb = unified.loc[
        (unified["feature_set"] == "enrolment")
        & (unified["model"] == "XGBoost")
    ]
    required_experiments = {
        "within_uci",
        "within_oulad",
        "zero_uci_to_oulad",
        "zero_oulad_to_uci",
        "pooled_to_uci",
        "pooled_to_oulad",
        "adapt_uci_to_oulad_20",
        "adapt_oulad_to_uci_20",
    }
    missing_experiments = required_experiments - set(
        unified_xgb["experiment"]
    )
    if missing_experiments:
        raise ValueError(
            "Missing unified-model experiments: "
            f"{sorted(missing_experiments)}"
        )
    if set(shared_schema["shared_feature"]) != {
        "age_scaled",
        "prior_preparation",
        "study_load",
        "male",
        "declared_support_need",
        "financial_stability",
    }:
        raise ValueError("The shared feature schema is incomplete.")

    required_comparisons = {
        "OULAD: zero-shot vs 20% local adaptation",
        "UCI: zero-shot vs 20% local adaptation",
        "OULAD: local-only vs pooled model",
        "UCI: local-only vs pooled model",
    }
    if set(unified_tests["comparison"]) != required_comparisons:
        raise ValueError("The unified paired-test table is incomplete.")
    if not unified_tests["permutation_p"].between(0, 1).all():
        raise ValueError("A paired-test p value is outside 0-1.")

    model_path = MODELS / "unified_enrolment_xgboost.joblib"
    contract_path = (
        MODELS / "unified_enrolment_feature_contract.json"
    )
    artifact = joblib.load(model_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_features = [
        "age_scaled",
        "prior_preparation",
        "study_load",
        "male",
        "declared_support_need",
        "financial_stability",
    ]
    if artifact["features"] != expected_features:
        raise ValueError("The saved model feature order is incorrect.")
    if contract["features"] != expected_features:
        raise ValueError("The saved feature contract is incorrect.")

    example = pd.read_csv(
        EXAMPLES / "new_school_enrolment_features.csv"
    )
    example_scores = artifact["pipeline"].predict_proba(
        example[expected_features]
    )[:, 1]
    if len(example_scores) != len(example):
        raise ValueError("The deployment scoring smoke test failed.")
    if not pd.Series(example_scores).between(0, 1).all():
        raise ValueError("The deployment model returned an invalid score.")

    dynamic_features = [
        "age_scaled",
        "prior_education_level",
        "study_load",
        "course_progress_ratio",
        "prior_academic_score",
        "previous_attempts",
        "male",
        "declared_support_need",
        "financial_stability",
        "assessment_completion_rate",
        "assessment_average_score",
        "assessment_score_available",
        "late_submission_rate",
        "active_day_rate",
        "days_since_last_activity_scaled",
        "recent_activity_rate",
    ]
    if dynamic_schema["feature"].tolist() != dynamic_features:
        raise ValueError("The dynamic feature schema is incorrect.")
    dynamic_overlap_columns = [
        column
        for column in dynamic_overlap.columns
        if column != "institution"
    ]
    if dynamic_overlap[dynamic_overlap_columns].to_numpy().max() != 0:
        raise ValueError("Dynamic-model student overlap was found.")

    dynamic_xgb = dynamic.loc[
        dynamic["model"].eq("XGBoost")
        & dynamic["training_scenario"].eq("pooled")
    ]
    expected_tests = {
        ("UCI_dataset", 0),
        ("OULAD_dataset", 0),
        ("OULAD_dataset", 35),
        ("OULAD_dataset", 60),
        ("OULAD_dataset", 75),
    }
    observed_tests = set(
        zip(
            dynamic_xgb["test_institution"],
            dynamic_xgb["snapshot_day"].astype(int),
        )
    )
    if observed_tests != expected_tests:
        raise ValueError("The dynamic pooled-model tests are incomplete.")
    bounded_columns = [
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
    ]
    if not dynamic_xgb[bounded_columns].apply(
        lambda values: values.between(0, 1).all()
    ).all():
        raise ValueError("A dynamic-model metric is outside 0-1.")

    expected_comparisons = {
        "UCI_dataset day 0: pooled minus local-only",
        *{
            f"OULAD_dataset day {day}: pooled minus local-only"
            for day in (0, 35, 60, 75)
        },
    }
    if set(dynamic_tests["comparison"]) != expected_comparisons:
        raise ValueError("The dynamic paired-test table is incomplete.")
    if not dynamic_tests["permutation_p"].between(0, 1).all():
        raise ValueError("A dynamic paired-test p value is outside 0-1.")
    shap_tests = set(
        zip(
            dynamic_shap["institution"],
            dynamic_shap["snapshot_day"].astype(int),
        )
    )
    if shap_tests != expected_tests:
        raise ValueError("The dynamic SHAP summary is incomplete.")
    if set(dynamic_shap["feature"]) != set(dynamic_features):
        raise ValueError("The dynamic SHAP feature list is incomplete.")

    dynamic_model_path = MODELS / "unified_dynamic_xgboost.joblib"
    dynamic_contract_path = (
        MODELS / "unified_dynamic_feature_contract.json"
    )
    dynamic_artifact = joblib.load(dynamic_model_path)
    dynamic_contract = json.loads(
        dynamic_contract_path.read_text(encoding="utf-8")
    )
    if dynamic_artifact["features"] != dynamic_features:
        raise ValueError("The dynamic model feature order is incorrect.")
    if dynamic_contract["features"] != dynamic_features:
        raise ValueError("The dynamic feature contract is incorrect.")

    dynamic_example = pd.read_csv(
        EXAMPLES / "new_school_dynamic_features.csv"
    )
    dynamic_scores = dynamic_artifact["pipeline"].predict_proba(
        dynamic_example[dynamic_features]
    )[:, 1]
    if len(dynamic_scores) != len(dynamic_example):
        raise ValueError("The dynamic scoring smoke test failed.")
    if not pd.Series(dynamic_scores).between(0, 1).all():
        raise ValueError("The dynamic model returned an invalid score.")

    print("Output check passed.")
    print(
        "Day-35 OULAD XGBoost ROC-AUC:",
        round(day_35.iloc[0]["roc_auc"], 3),
    )
    print(
        "Day-35 OULAD XGBoost AP:",
        round(day_35.iloc[0]["average_precision"], 3),
    )
    pooled_uci = unified_xgb.loc[
        unified_xgb["experiment"].eq("pooled_to_uci")
    ].iloc[0]
    pooled_oulad = unified_xgb.loc[
        unified_xgb["experiment"].eq("pooled_to_oulad")
    ].iloc[0]
    print(
        "Pooled unified model ROC-AUC (UCI):",
        round(pooled_uci["roc_auc"], 3),
    )
    print(
        "Pooled unified model ROC-AUC (OULAD):",
        round(pooled_oulad["roc_auc"], 3),
    )
    dynamic_day35 = dynamic_xgb.loc[
        dynamic_xgb["test_institution"].eq("OULAD_dataset")
        & dynamic_xgb["snapshot_day"].eq(35)
    ].iloc[0]
    print(
        "Portable dynamic model day-35 ROC-AUC:",
        round(dynamic_day35["roc_auc"], 3),
    )
    print(
        "Portable dynamic model day-35 F1:",
        round(dynamic_day35["f1"], 3),
    )


if __name__ == "__main__":
    main()
