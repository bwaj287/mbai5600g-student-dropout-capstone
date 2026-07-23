import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Reuse the same pipeline from Milestone 5 instead of copying it here.
ROOT = Path(__file__).resolve().parent
CAPSTONE = ROOT.parents[1]
MILESTONE5 = ROOT.parent / "milestone5"
sys.path.insert(0, str(MILESTONE5))

import milestone5_model_development as m5


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
M5_METRICS = MILESTONE5 / "results" / "tuned_model_metrics.json"

RANDOM_STATE = 42
REPEATED_SPLIT_SEEDS = [7, 21, 42, 84, 126]

FINAL_MODELS = [
    {
        "task": "uci_binary_early",
        "model": "logistic_regression",
        "label": "UCI early warning - Logistic Regression",
        "setting": "student-record early warning",
    },
    {
        "task": "oulad_binary_early",
        "model": "xgboost",
        "label": "OULAD early warning - XGBoost",
        "setting": "online-learning early warning",
    },
]

ERROR_FEATURES = {
    "uci_binary_early": [
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (enrolled)",
        "Tuition fees up to date",
        "Age at enrollment",
    ],
    "oulad_binary_early": [
        "assessment_submission_ratio_early",
        "assessment_score_max_early",
        "assessment_weighted_score_ratio_early",
        "vle_total_clicks_early",
        "vle_active_days_early",
    ],
}

SEGMENT_COLUMNS = {
    "uci_binary_early": [
        "Gender",
        "Tuition fees up to date",
        "Debtor",
        "Scholarship holder",
    ],
    "oulad_binary_early": [
        "gender",
        "disability",
        "age_band",
        "highest_education",
    ],
}

ALERT_CAPACITY_PER_WEEK = 25
INTERVENTION_COST = 100
RETAINED_STUDENT_VALUE = 6000
RETENTION_EFFECT = 0.10


def print_step(number, title):
    print("\n" + "=" * 70)
    print(f"Step {number} - {title}")
    print("=" * 70)


def load_selected_settings():
    milestone5_results = json.loads(M5_METRICS.read_text())
    settings = {}

    for item in FINAL_MODELS:
        task = item["task"]
        model = item["model"]
        model_results = milestone5_results["tasks"][task]["models"][model]
        settings[task] = {
            **item,
            "params": model_results["best_params"],
            "threshold": model_results["selected_threshold"],
        }
    return settings


def load_task_data(task):
    config = m5.TASKS[task]
    frame, features, target, _, _, _ = m5.prepare_task_data(task, config)
    categorical, numeric = m5.split_feature_types(features)
    return frame, features, target, categorical, numeric


def split_data(features, target, seed):
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=seed,
        stratify=target,
    )


def error_labels(actual, predicted):
    labels = np.full(len(actual), "", dtype=object)
    labels[(actual == 0) & (predicted == 0)] = "TN"
    labels[(actual == 0) & (predicted == 1)] = "FP"
    labels[(actual == 1) & (predicted == 0)] = "FN"
    labels[(actual == 1) & (predicted == 1)] = "TP"
    return labels


def train_final_model(task, settings):
    frame, features, target, categorical, numeric = load_task_data(task)
    x_train, x_test, y_train, y_test = split_data(
        features,
        target,
        RANDOM_STATE,
    )

    start_fit = time.perf_counter()
    pipeline = m5.build_pipeline(
        settings["model"],
        "binary",
        settings["params"],
        categorical,
        numeric,
        y_train,
        2,
    )
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start_fit

    start_predict = time.perf_counter()
    risk_score = pipeline.predict_proba(x_test)[:, 1]
    prediction = (risk_score >= settings["threshold"]).astype(int)
    predict_seconds = time.perf_counter() - start_predict

    metrics = m5.metrics_for_predictions(
        "binary",
        y_test,
        prediction,
        risk_score,
    )
    matrix = confusion_matrix(y_test, prediction)
    tn, fp, fn, tp = matrix.ravel()

    prediction_frame = frame.loc[x_test.index].copy()
    prediction_frame["actual_attrition"] = y_test.to_numpy()
    prediction_frame["predicted_attrition"] = prediction
    prediction_frame["risk_score"] = risk_score
    prediction_frame["error_type"] = error_labels(
        y_test.to_numpy(),
        prediction,
    )

    prediction_path = RESULTS / f"{task}_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    validation_row = {
        "task": task,
        "model": settings["model"],
        "model_label": settings["label"],
        "setting": settings["setting"],
        "rows": len(frame),
        "test_rows": len(x_test),
        "selected_threshold": settings["threshold"],
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "prediction_path": str(
            prediction_path.relative_to(CAPSTONE.parent)
        ),
        **metrics,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    return {
        "validation_row": validation_row,
        "frame": frame,
        "features": features,
        "target": target,
        "categorical": categorical,
        "numeric": numeric,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "pipeline": pipeline,
        "risk_score": risk_score,
        "prediction": prediction,
        "prediction_frame": prediction_frame,
    }


def missing_value_test(model_data, settings):
    # Seed 142 is RANDOM_STATE + 100.
    rng = np.random.default_rng(RANDOM_STATE + 100)
    changed_data = model_data["x_test"].copy()

    # Pandas string columns use pd.NA by default. Converting them to object
    # lets the same np.nan mask work for numeric and categorical columns.
    for column in model_data["categorical"]:
        changed_data[column] = changed_data[column].astype(object)

    missing_mask = rng.random(changed_data.shape) < 0.10
    changed_array = changed_data.to_numpy(dtype=object)
    changed_array[missing_mask] = np.nan
    changed_data = pd.DataFrame(
        changed_array,
        index=model_data["x_test"].index,
        columns=model_data["x_test"].columns,
    )
    for column in model_data["numeric"]:
        changed_data[column] = pd.to_numeric(
            changed_data[column],
            errors="coerce",
        )

    score = model_data["pipeline"].predict_proba(changed_data)[:, 1]
    prediction = (score >= settings["threshold"]).astype(int)
    return score, prediction


def numeric_noise_test(model_data, settings):
    # Seed 242 is RANDOM_STATE + 200.
    rng = np.random.default_rng(RANDOM_STATE + 200)
    changed_data = model_data["x_test"].copy()

    for column in model_data["numeric"]:
        train_values = pd.to_numeric(
            model_data["x_train"][column],
            errors="coerce",
        )
        test_values = pd.to_numeric(
            changed_data[column],
            errors="coerce",
        )
        noise_size = train_values.std() * 0.05
        if pd.isna(noise_size) or noise_size == 0:
            continue
        changed_data[column] = test_values + rng.normal(
            0,
            noise_size,
            len(changed_data),
        )

    score = model_data["pipeline"].predict_proba(changed_data)[:, 1]
    prediction = (score >= settings["threshold"]).astype(int)
    return score, prediction


def robustness_row(task, settings, model_data, scenario, score, prediction):
    original_score = model_data["risk_score"]
    original_prediction = model_data["prediction"]
    metrics = m5.metrics_for_predictions(
        "binary",
        model_data["y_test"],
        prediction,
        score,
    )
    return {
        "task": task,
        "model": settings["model"],
        "model_label": settings["label"],
        "scenario": scenario,
        "changed_prediction_rate": float(
            np.mean(prediction != original_prediction)
        ),
        "mean_abs_score_shift": float(
            np.mean(np.abs(score - original_score))
        ),
        "score_correlation_with_original": float(
            np.corrcoef(original_score, score)[0, 1]
        ),
        **metrics,
    }


def run_robustness_tests(task, settings, model_data):
    rows = [
        robustness_row(
            task,
            settings,
            model_data,
            "original",
            model_data["risk_score"],
            model_data["prediction"],
        )
    ]

    missing_score, missing_prediction = missing_value_test(
        model_data,
        settings,
    )
    rows.append(
        robustness_row(
            task,
            settings,
            model_data,
            "missing_10pct",
            missing_score,
            missing_prediction,
        )
    )

    noise_score, noise_prediction = numeric_noise_test(
        model_data,
        settings,
    )
    rows.append(
        robustness_row(
            task,
            settings,
            model_data,
            "numeric_noise_5pct",
            noise_score,
            noise_prediction,
        )
    )
    return rows


def run_repeated_splits(task, settings, model_data):
    rows = []
    for seed in REPEATED_SPLIT_SEEDS:
        x_train, x_test, y_train, y_test = split_data(
            model_data["features"],
            model_data["target"],
            seed,
        )
        pipeline = m5.build_pipeline(
            settings["model"],
            "binary",
            settings["params"],
            model_data["categorical"],
            model_data["numeric"],
            y_train,
            2,
        )
        pipeline.fit(x_train, y_train)
        score = pipeline.predict_proba(x_test)[:, 1]
        prediction = (score >= settings["threshold"]).astype(int)
        metrics = m5.metrics_for_predictions(
            "binary",
            y_test,
            prediction,
            score,
        )
        rows.append(
            {
                "task": task,
                "model": settings["model"],
                "model_label": settings["label"],
                "random_state": seed,
                "holdout_rows": len(x_test),
                **metrics,
            }
        )
    return rows


def summarize_repeated_splits(resampling_frame):
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]
    rows = []

    for model_label, group in resampling_frame.groupby("model_label"):
        row = {
            "model_label": model_label,
            "folds": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_errors(task, settings, prediction_frame):
    summary_rows = []
    for error_type, group in prediction_frame.groupby("error_type"):
        summary_rows.append(
            {
                "task": task,
                "model": settings["model"],
                "model_label": settings["label"],
                "error_type": error_type,
                "count": len(group),
                "share_of_test": len(group) / len(prediction_frame),
                "mean_risk_score": group["risk_score"].mean(),
            }
        )

    profile_rows = []
    for feature in ERROR_FEATURES[task]:
        for error_type, group in prediction_frame.groupby("error_type"):
            values = pd.to_numeric(group[feature], errors="coerce")
            profile_rows.append(
                {
                    "task": task,
                    "feature": feature,
                    "error_type": error_type,
                    "count": int(values.notna().sum()),
                    "mean": values.mean(),
                    "median": values.median(),
                }
            )
    return summary_rows, profile_rows


def analyze_segments(task, prediction_frame):
    rows = []
    for column in SEGMENT_COLUMNS[task]:
        for segment_value, group in prediction_frame.groupby(
            column,
            dropna=False,
        ):
            actual = group["actual_attrition"]
            predicted = group["predicted_attrition"]
            tn, fp, fn, tp = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ).ravel()
            rows.append(
                {
                    "task": task,
                    "segment_column": column,
                    "segment_value": segment_value,
                    "rows": len(group),
                    "actual_attrition_rate": actual.mean(),
                    "alert_rate": predicted.mean(),
                    "precision": precision_score(
                        actual,
                        predicted,
                        zero_division=0,
                    ),
                    "recall": recall_score(
                        actual,
                        predicted,
                        zero_division=0,
                    ),
                    "f1": f1_score(
                        actual,
                        predicted,
                        zero_division=0,
                    ),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )
    return rows


def calculate_business_impact(validation_row):
    test_rows = validation_row["test_rows"]
    alerts = (
        validation_row["true_positive"]
        + validation_row["false_positive"]
    )
    alerts_per_1000 = alerts / test_rows * 1000
    found_per_1000 = validation_row["true_positive"] / test_rows * 1000
    missed_per_1000 = validation_row["false_negative"] / test_rows * 1000
    false_alerts_per_1000 = (
        validation_row["false_positive"] / test_rows * 1000
    )

    estimated_value = (
        found_per_1000 * RETENTION_EFFECT * RETAINED_STUDENT_VALUE
    )
    estimated_cost = alerts_per_1000 * INTERVENTION_COST
    break_even_students = (
        estimated_cost / RETAINED_STUDENT_VALUE
    )

    return {
        "task": validation_row["task"],
        "model_label": validation_row["model_label"],
        "test_rows": test_rows,
        "actual_attrition_cases": (
            validation_row["true_positive"]
            + validation_row["false_negative"]
        ),
        "alerts": alerts,
        "alerts_per_1000": alerts_per_1000,
        "true_at_risk_found_per_1000": found_per_1000,
        "missed_at_risk_per_1000": missed_per_1000,
        "false_alerts_per_1000": false_alerts_per_1000,
        "advisor_weeks_per_1000": (
            alerts_per_1000 / ALERT_CAPACITY_PER_WEEK
        ),
        "precision": validation_row["precision"],
        "recall": validation_row["recall"],
        "f1": validation_row["f1"],
        "intervention_cost_assumption": INTERVENTION_COST,
        "retained_student_value_assumption": RETAINED_STUDENT_VALUE,
        "retention_effect_assumption": RETENTION_EFFECT,
        "scenario_net_value_per_1000": estimated_value - estimated_cost,
        "break_even_retained_students_per_1000": break_even_students,
        "break_even_capture_rate_of_found_at_risk": (
            break_even_students / found_per_1000
        ),
    }


def save_figures(
    validation_frame,
    robustness_frame,
    error_frame,
    business_frame,
):
    labels = validation_frame["model_label"].str.replace(
        " early warning - ",
        "\n",
        regex=False,
    )

    # Final model metrics
    chart_data = validation_frame.set_index(labels)[
        ["precision", "recall", "f1", "roc_auc"]
    ]
    ax = chart_data.plot(kind="bar", figsize=(9, 5))
    ax.set_title("Final Model Validation")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURES / "final_model_validation_metrics.png",
        dpi=180,
    )
    plt.close()

    # Robustness comparison
    robustness_plot = robustness_frame.pivot(
        index="model_label",
        columns="scenario",
        values="f1",
    )
    ax = robustness_plot.plot(kind="bar", figsize=(9, 5))
    ax.set_title("F1 Under Robustness Tests")
    ax.set_ylabel("F1")
    ax.set_ylim(0.65, 0.85)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(
        FIGURES / "robustness_sensitivity_f1.png",
        dpi=180,
    )
    plt.close()

    # Error counts
    error_plot = error_frame.pivot(
        index="model_label",
        columns="error_type",
        values="count",
    )
    ax = error_plot[["FP", "FN"]].plot(kind="bar", figsize=(8, 5))
    ax.set_title("False Positive and False Negative Counts")
    ax.set_ylabel("Students in held-out test set")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "error_type_counts.png", dpi=180)
    plt.close()

    # Business impact per 1,000
    impact_columns = [
        "alerts_per_1000",
        "true_at_risk_found_per_1000",
        "false_alerts_per_1000",
        "missed_at_risk_per_1000",
    ]
    impact_plot = business_frame.set_index("model_label")[impact_columns]
    ax = impact_plot.plot(kind="bar", figsize=(10, 5))
    ax.set_title("Illustrative Operational Impact per 1,000 Students")
    ax.set_ylabel("Students")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "business_impact_per_1000.png", dpi=180)
    plt.close()


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    settings_by_task = load_selected_settings()

    print_step(1, "Validate the Two Final Models")
    validation_rows = []
    saved_model_data = {}

    for task, settings in settings_by_task.items():
        print("\nRunning:", settings["label"])
        model_data = train_final_model(task, settings)
        saved_model_data[task] = model_data
        validation_rows.append(model_data["validation_row"])
        print(
            "Test F1:",
            round(model_data["validation_row"]["f1"], 4),
        )
        print(
            "Test ROC-AUC:",
            round(model_data["validation_row"]["roc_auc"], 4),
        )

    validation_frame = pd.DataFrame(validation_rows)
    validation_frame.to_csv(
        RESULTS / "final_model_validation.csv",
        index=False,
    )

    print_step(2, "Run Missingness and Numeric Noise Tests")
    robustness_rows = []
    for task, settings in settings_by_task.items():
        robustness_rows.extend(
            run_robustness_tests(
                task,
                settings,
                saved_model_data[task],
            )
        )
    robustness_frame = pd.DataFrame(robustness_rows)
    robustness_frame.to_csv(
        RESULTS / "robustness_sensitivity.csv",
        index=False,
    )
    print(
        robustness_frame[
            ["model_label", "scenario", "f1", "changed_prediction_rate"]
        ].to_string(index=False)
    )

    print_step(3, "Check Stability Across Five Hold-out Splits")
    resampling_rows = []
    for task, settings in settings_by_task.items():
        resampling_rows.extend(
            run_repeated_splits(
                task,
                settings,
                saved_model_data[task],
            )
        )
    resampling_frame = pd.DataFrame(resampling_rows)
    resampling_frame.to_csv(
        RESULTS / "resampling_stability.csv",
        index=False,
    )
    resampling_summary = summarize_repeated_splits(resampling_frame)
    resampling_summary.to_csv(
        RESULTS / "resampling_stability_summary.csv",
        index=False,
    )
    print(
        resampling_summary[
            ["model_label", "f1_mean", "f1_std", "roc_auc_mean"]
        ].to_string(index=False)
    )

    print_step(4, "Review Errors and Student Segments")
    error_rows = []
    profile_rows = []
    segment_rows = []
    for task, settings in settings_by_task.items():
        prediction_frame = saved_model_data[task]["prediction_frame"]
        task_errors, task_profiles = analyze_errors(
            task,
            settings,
            prediction_frame,
        )
        error_rows.extend(task_errors)
        profile_rows.extend(task_profiles)
        segment_rows.extend(
            analyze_segments(task, prediction_frame)
        )

    error_frame = pd.DataFrame(error_rows)
    profile_frame = pd.DataFrame(profile_rows)
    segment_frame = pd.DataFrame(segment_rows)
    error_frame.to_csv(RESULTS / "error_summary.csv", index=False)
    profile_frame.to_csv(
        RESULTS / "error_feature_profile.csv",
        index=False,
    )
    segment_frame.to_csv(
        RESULTS / "segment_performance.csv",
        index=False,
    )
    print(
        error_frame[
            ["model_label", "error_type", "count", "share_of_test"]
        ].to_string(index=False)
    )

    print_step(5, "Estimate Operational Impact per 1,000 Students")
    business_rows = [
        calculate_business_impact(row)
        for row in validation_rows
    ]
    business_frame = pd.DataFrame(business_rows)
    business_frame.to_csv(
        RESULTS / "business_impact.csv",
        index=False,
    )
    print(
        business_frame[
            [
                "model_label",
                "alerts_per_1000",
                "true_at_risk_found_per_1000",
                "false_alerts_per_1000",
                "missed_at_risk_per_1000",
            ]
        ].round(1).to_string(index=False)
    )

    print_step(6, "Save Figures and Summary")
    save_figures(
        validation_frame,
        robustness_frame,
        error_frame,
        business_frame,
    )

    summary = {
        "final_models": FINAL_MODELS,
        "results": {
            "final_model_validation": "capstone/work/milestone6/results/final_model_validation.csv",
            "robustness_sensitivity": "capstone/work/milestone6/results/robustness_sensitivity.csv",
            "resampling_stability": "capstone/work/milestone6/results/resampling_stability.csv",
            "resampling_stability_summary": "capstone/work/milestone6/results/resampling_stability_summary.csv",
            "error_summary": "capstone/work/milestone6/results/error_summary.csv",
            "error_feature_profile": "capstone/work/milestone6/results/error_feature_profile.csv",
            "segment_performance": "capstone/work/milestone6/results/segment_performance.csv",
            "business_impact": "capstone/work/milestone6/results/business_impact.csv",
        },
        "figures": {
            "validation_metrics": "capstone/work/milestone6/figures/final_model_validation_metrics.png",
            "robustness_sensitivity": "capstone/work/milestone6/figures/robustness_sensitivity_f1.png",
            "business_impact": "capstone/work/milestone6/figures/business_impact_per_1000.png",
            "error_counts": "capstone/work/milestone6/figures/error_type_counts.png",
        },
        "business_assumptions": {
            "advisor_alert_capacity_per_week": ALERT_CAPACITY_PER_WEEK,
            "intervention_cost_per_alert": INTERVENTION_COST,
            "retained_student_value": RETAINED_STUDENT_VALUE,
            "illustrative_retention_effect": RETENTION_EFFECT,
        },
    }
    (RESULTS / "milestone6_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\nMilestone 6 outputs were saved in:", RESULTS)


if __name__ == "__main__":
    main()
