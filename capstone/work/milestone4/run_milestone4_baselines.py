from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
RANDOM_STATE = 42

UCI_CONTINUOUS_COLUMNS = {
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}


TASKS = {
    "uci_multiclass": {
        "path": DATA / "uci_multiclass_model_ready.csv",
        "target": "Target",
        "drop_columns": ["is_attrition"],
        "problem_type": "multiclass",
        "primary_metric": "macro_f1",
    },
    "uci_binary_early": {
        "path": DATA / "uci_binary_early_model_ready.csv",
        "target": "is_attrition",
        "drop_columns": ["Target"],
        "problem_type": "binary",
        "primary_metric": "f1",
    },
    "oulad_binary_early": {
        "path": DATA / "oulad_binary_early_model_ready.csv",
        "target": "is_attrition",
        "drop_columns": ["final_result", "id_student"],
        "problem_type": "binary",
        "primary_metric": "f1",
    },
}


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in frame.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def build_pipeline(problem_type: str, categorical_columns: list[str], numeric_columns: list[str], model_name: str) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_predictions(problem_type: str, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, object]:
    if problem_type == "multiclass":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def save_confusion_matrix(task_name: str, model_name: str, y_true: pd.Series, y_pred: np.ndarray, labels: list[object]) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{task_name} - {model_name}")

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
            )

    fig.tight_layout()
    out_path = FIGURES / f"{task_name}_{model_name}_confusion_matrix.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(out_path)


def extract_feature_importance(fitted_pipeline: Pipeline, problem_type: str) -> pd.DataFrame:
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        coefficients = classifier.coef_
        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        else:
            importances = np.abs(coefficients).mean(axis=0)

    importance_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)
    importance_frame["problem_type"] = problem_type
    return importance_frame


def save_feature_plot(task_name: str, model_name: str, importance_frame: pd.DataFrame) -> str:
    top_features = importance_frame.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f8f")
    ax.set_title(f"Top Features: {task_name} - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    out_path = FIGURES / f"{task_name}_{model_name}_top_features.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(out_path)


def run_task(task_name: str, config: dict[str, object]) -> dict[str, object]:
    frame = pd.read_csv(config["path"])
    target_name = config["target"]
    drop_columns = set(config["drop_columns"]) | {target_name}
    feature_frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns])

    if task_name.startswith("uci_"):
        uci_categorical_columns = [
            column for column in feature_frame.columns if column not in UCI_CONTINUOUS_COLUMNS
        ]
        for column in uci_categorical_columns:
            feature_frame[column] = feature_frame[column].astype("string")

    target_series = frame[target_name]
    categorical_columns, numeric_columns = split_feature_types(feature_frame)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target_series,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target_series,
    )

    model_results: dict[str, object] = {}
    summary_rows: list[dict[str, object]] = []
    primary_metric = config["primary_metric"]

    for model_name in ("logistic_regression", "random_forest"):
        pipeline = build_pipeline(
            config["problem_type"], categorical_columns, numeric_columns, model_name
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)

        if config["problem_type"] == "binary":
            y_score = pipeline.predict_proba(x_test)[:, 1]
            labels = [0, 1]
        else:
            y_score = None
            labels = sorted(target_series.unique().tolist())

        metrics = evaluate_predictions(config["problem_type"], y_test, y_pred, y_score)
        confusion_matrix_path = save_confusion_matrix(
            task_name, model_name, y_test, y_pred, labels
        )
        importance_frame = extract_feature_importance(pipeline, config["problem_type"])
        importance_csv_path = RESULTS / f"{task_name}_{model_name}_feature_importance.csv"
        importance_frame.to_csv(importance_csv_path, index=False)
        feature_plot_path = save_feature_plot(task_name, model_name, importance_frame)

        model_results[model_name] = {
            "metrics": metrics,
            "confusion_matrix_path": confusion_matrix_path,
            "feature_importance_path": rel(importance_csv_path),
            "feature_plot_path": feature_plot_path,
        }

        summary_row = {"task": task_name, "model": model_name}
        for metric_name, metric_value in metrics.items():
            if metric_name == "classification_report":
                continue
            summary_row[metric_name] = metric_value
        summary_rows.append(summary_row)

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["metrics"][primary_metric],
    )
    best_metrics = model_results[best_model_name]["metrics"]

    return {
        "task": task_name,
        "problem_type": config["problem_type"],
        "target": target_name,
        "row_count": int(frame.shape[0]),
        "feature_count": int(feature_frame.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "class_distribution": target_series.value_counts().to_dict(),
        "best_model": best_model_name,
        "best_metrics": {
            key: value
            for key, value in best_metrics.items()
            if key != "classification_report"
        },
        "models": model_results,
        "summary_rows": summary_rows,
    }


def main() -> None:
    ensure_dirs()
    task_outputs = {}
    summary_rows = []
    for task_name, config in TASKS.items():
        task_output = run_task(task_name, config)
        task_outputs[task_name] = task_output
        summary_rows.extend(task_output["summary_rows"])

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(RESULTS / "baseline_comparison.csv", index=False)

    payload = {
        "random_state": RANDOM_STATE,
        "tasks": task_outputs,
    }
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
