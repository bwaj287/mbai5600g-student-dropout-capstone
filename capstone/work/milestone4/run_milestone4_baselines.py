from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime with a clear message
    XGBClassifier = None


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SHARE_OF_TRAINVAL = 0.25
CV_FOLDS = 5
IQR_MULTIPLIER = 1.5
OUTLIER_CLIP_LOWER = 0.01
OUTLIER_CLIP_UPPER = 0.99

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

MODEL_ORDER = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
)

ARTICLE_MODEL_NAMES = (
    "decision_tree",
    "random_forest",
    "gradient_boosting",
    "xgboost",
)

ARTICLE_BASELINE_REFERENCE = {
    "citation_key": "Islam2025",
    "best_reported_model": "xgboost",
    "reported_accuracy": 0.83,
    "reported_problem": "multiclass student academic performance prediction on the UCI dataset",
    "reported_xai_methods": ["SHAP", "Shapash", "Eli5", "LIME"],
}

TASKS = {
    "uci_multiclass": {
        "display_name": "UCI multiclass reproduction",
        "path": DATA / "uci_multiclass_model_ready.csv",
        "target": "Target",
        "drop_columns": ["is_attrition"],
        "problem_type": "multiclass",
        "primary_metric": "macro_f1",
        "role": "baseline_reproduction",
        "class_names": ["Dropout", "Enrolled", "Graduate"],
    },
    "uci_binary_early": {
        "display_name": "UCI binary early warning",
        "path": DATA / "uci_binary_early_model_ready.csv",
        "target": "is_attrition",
        "drop_columns": ["Target"],
        "problem_type": "binary",
        "primary_metric": "f1",
        "role": "early_warning_extension",
        "class_names": ["Graduate_or_Enrolled", "Dropout"],
    },
    "oulad_binary_early": {
        "display_name": "OULAD binary early warning",
        "path": DATA / "oulad_binary_early_model_ready.csv",
        "target": "is_attrition",
        "drop_columns": ["final_result", "id_student"],
        "problem_type": "binary",
        "primary_metric": "f1",
        "role": "early_warning_extension",
        "class_names": ["Pass_Distinction_Fail", "Withdrawn"],
    },
}


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile: float = OUTLIER_CLIP_LOWER, upper_quantile: float = OUTLIER_CLIP_UPPER):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x, y=None):
        x_array = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(x_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(x_array, self.upper_quantile, axis=0)
        return self

    def transform(self, x):
        x_array = np.asarray(x, dtype=float)
        return np.clip(x_array, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return np.asarray(input_features, dtype=object)


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in frame.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def get_model_config(model_name: str, problem_type: str, class_count: int | None = None) -> dict[str, object]:
    if model_name == "logistic_regression":
        return {
            "max_iter": 3000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": RANDOM_STATE,
        }
    if model_name == "decision_tree":
        return {
            "criterion": "gini",
            "max_depth": 8,
            "min_samples_leaf": 10,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
    if model_name == "random_forest":
        return {
            "n_estimators": 400,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
    if model_name == "gradient_boosting":
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 0.9,
            "random_state": RANDOM_STATE,
        }
    if model_name == "xgboost":
        if problem_type == "multiclass":
            objective = "multi:softprob"
            eval_metric = "mlogloss"
            num_class = class_count
        else:
            objective = "binary:logistic"
            eval_metric = "logloss"
            num_class = None
        config = {
            "n_estimators": 250,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "objective": objective,
            "eval_metric": eval_metric,
            "verbosity": 0,
        }
        if num_class is not None:
            config["num_class"] = num_class
        return config
    raise ValueError(f"Unsupported model: {model_name}")


def build_estimator(model_name: str, problem_type: str, class_count: int | None = None):
    config = get_model_config(model_name, problem_type, class_count)
    if model_name == "logistic_regression":
        return LogisticRegression(**config)
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**config)
    if model_name == "random_forest":
        return RandomForestClassifier(**config)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(**config)
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for the article-aligned baseline. "
                "Run with `uv run --with xgboost ...`."
            )
        return XGBClassifier(**config)
    raise ValueError(f"Unsupported model: {model_name}")


def build_pipeline(
    problem_type: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    model_name: str,
    class_count: int | None = None,
) -> Pipeline:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper()),
    ]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
            ("numeric", Pipeline(steps=numeric_steps), numeric_columns),
        ]
    )

    classifier = build_estimator(model_name, problem_type, class_count)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_label_values(target_series: pd.Series) -> list[int]:
    return sorted(pd.Series(target_series).dropna().unique().tolist())


def evaluate_predictions(
    problem_type: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
    labels: list[int],
    label_names: list[str],
) -> dict[str, object]:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    if problem_type == "multiclass":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "classification_report": report,
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "classification_report": report,
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def save_confusion_matrix(
    task_name: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[int],
    display_labels: list[str],
) -> str:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_yticklabels(display_labels)
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


def get_prediction_scores(problem_type: str, fitted_pipeline: Pipeline, feature_frame: pd.DataFrame) -> np.ndarray | None:
    if problem_type != "binary" or not hasattr(fitted_pipeline, "predict_proba"):
        return None
    probabilities = fitted_pipeline.predict_proba(feature_frame)
    return probabilities[:, 1]


def summarize_outliers(frame: pd.DataFrame, numeric_columns: list[str]) -> dict[str, object]:
    if not numeric_columns:
        return {
            "method": "IQR",
            "iqr_multiplier": IQR_MULTIPLIER,
            "cell_outlier_count": 0,
            "features_with_outliers": 0,
            "top_features": [],
        }

    numeric_frame = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    q1 = numeric_frame.quantile(0.25)
    q3 = numeric_frame.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (IQR_MULTIPLIER * iqr)
    upper = q3 + (IQR_MULTIPLIER * iqr)
    outlier_mask = numeric_frame.lt(lower) | numeric_frame.gt(upper)
    outlier_counts = outlier_mask.sum().sort_values(ascending=False)
    outlier_rates = (outlier_mask.mean() * 100).round(2)
    top_features = []
    for column in outlier_counts.index:
        count = int(outlier_counts[column])
        if count == 0:
            continue
        top_features.append(
            {
                "feature": column,
                "count": count,
                "pct_rows": float(outlier_rates[column]),
            }
        )
        if len(top_features) == 10:
            break

    return {
        "method": "IQR",
        "iqr_multiplier": IQR_MULTIPLIER,
        "cell_outlier_count": int(outlier_mask.sum().sum()),
        "features_with_outliers": int((outlier_counts > 0).sum()),
        "top_features": top_features,
        "treatment": {
            "strategy": "winsorize_clip",
            "lower_quantile": OUTLIER_CLIP_LOWER,
            "upper_quantile": OUTLIER_CLIP_UPPER,
            "rationale": (
                "Retain extreme but plausible student behavior while reducing the influence of very large numeric values "
                "on linear and boosting baselines."
            ),
        },
    }


def build_cv_scoring(problem_type: str) -> dict[str, str]:
    if problem_type == "multiclass":
        return {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
            "weighted_f1": "f1_weighted",
        }
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }


def run_cross_validation_summary(
    problem_type: str,
    categorical_columns: list[str],
    numeric_columns: list[str],
    model_name: str,
    class_count: int | None,
    x_train_val: pd.DataFrame,
    y_train_val: pd.Series,
) -> dict[str, object]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(
        problem_type,
        categorical_columns,
        numeric_columns,
        model_name,
        class_count,
    )
    scoring = build_cv_scoring(problem_type)
    cv_scores = cross_validate(
        pipeline,
        x_train_val,
        y_train_val,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )
    summary = {}
    for metric_name in scoring:
        values = cv_scores[f"test_{metric_name}"]
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return {
        "folds": CV_FOLDS,
        "metrics": summary,
    }


def make_summary_row(
    task_name: str,
    model_name: str,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    selected_model_name: str,
    cv_summary: dict[str, object] | None,
) -> dict[str, object]:
    row = {
        "task": task_name,
        "model": model_name,
        "selected_by_validation": model_name == selected_model_name,
    }
    for prefix, metrics in (("validation", validation_metrics), ("test", test_metrics)):
        for metric_name, metric_value in metrics.items():
            if metric_name == "classification_report":
                continue
            row[f"{prefix}_{metric_name}"] = metric_value
    if cv_summary and model_name == selected_model_name:
        for metric_name, metric_stats in cv_summary["metrics"].items():
            row[f"cv_{metric_name}_mean"] = metric_stats["mean"]
            row[f"cv_{metric_name}_std"] = metric_stats["std"]
    return row


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

    target_series_raw = frame[target_name]
    label_names = config["class_names"]
    label_encoder = None
    if config["problem_type"] == "multiclass":
        label_encoder = LabelEncoder()
        target_series = pd.Series(
            label_encoder.fit_transform(target_series_raw),
            index=target_series_raw.index,
            name=target_name,
        )
    else:
        target_series = target_series_raw.astype(int)

    categorical_columns, numeric_columns = split_feature_types(feature_frame)
    class_count = len(get_label_values(target_series))

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        feature_frame,
        target_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target_series,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VALIDATION_SHARE_OF_TRAINVAL,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    outlier_summary = summarize_outliers(x_train, numeric_columns)
    labels = get_label_values(target_series)

    model_results: dict[str, object] = {}
    final_test_metrics_by_model: dict[str, dict[str, object]] = {}
    final_validation_metrics_by_model: dict[str, dict[str, object]] = {}

    for model_name in MODEL_ORDER:
        validation_pipeline = build_pipeline(
            config["problem_type"],
            categorical_columns,
            numeric_columns,
            model_name,
            class_count,
        )
        validation_pipeline.fit(x_train, y_train)
        val_predictions = validation_pipeline.predict(x_val)
        val_scores = get_prediction_scores(config["problem_type"], validation_pipeline, x_val)
        validation_metrics = evaluate_predictions(
            config["problem_type"],
            y_val,
            val_predictions,
            val_scores,
            labels,
            label_names,
        )
        final_validation_metrics_by_model[model_name] = validation_metrics

        final_pipeline = build_pipeline(
            config["problem_type"],
            categorical_columns,
            numeric_columns,
            model_name,
            class_count,
        )
        final_pipeline.fit(x_train_val, y_train_val)
        test_predictions = final_pipeline.predict(x_test)
        test_scores = get_prediction_scores(config["problem_type"], final_pipeline, x_test)
        test_metrics = evaluate_predictions(
            config["problem_type"],
            y_test,
            test_predictions,
            test_scores,
            labels,
            label_names,
        )
        final_test_metrics_by_model[model_name] = test_metrics

        confusion_matrix_path = save_confusion_matrix(
            task_name,
            model_name,
            y_test,
            test_predictions,
            labels,
            label_names,
        )
        importance_frame = extract_feature_importance(final_pipeline, config["problem_type"])
        importance_csv_path = RESULTS / f"{task_name}_{model_name}_feature_importance.csv"
        importance_frame.to_csv(importance_csv_path, index=False)
        feature_plot_path = save_feature_plot(task_name, model_name, importance_frame)

        model_results[model_name] = {
            "model_role": (
                "article_baseline"
                if model_name in ARTICLE_MODEL_NAMES
                else "reference_baseline"
            ),
            "model_config": get_model_config(model_name, config["problem_type"], class_count),
            "preprocessing": {
                "categorical_imputer": "most_frequent",
                "categorical_encoder": "one_hot",
                "numeric_imputer": "median",
                "numeric_outlier_treatment": f"clip_{OUTLIER_CLIP_LOWER:.0%}_{OUTLIER_CLIP_UPPER:.0%}_quantiles",
                "numeric_scaling": "standard_scaler" if model_name == "logistic_regression" else "not_applied",
            },
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "confusion_matrix_path": confusion_matrix_path,
            "feature_importance_path": rel(importance_csv_path),
            "feature_plot_path": feature_plot_path,
        }

    primary_metric = config["primary_metric"]
    best_model_name = max(
        MODEL_ORDER,
        key=lambda name: final_validation_metrics_by_model[name][primary_metric],
    )
    cv_summary = run_cross_validation_summary(
        config["problem_type"],
        categorical_columns,
        numeric_columns,
        best_model_name,
        class_count,
        x_train_val,
        y_train_val,
    )

    summary_rows = [
        make_summary_row(
            task_name,
            model_name,
            final_validation_metrics_by_model[model_name],
            final_test_metrics_by_model[model_name],
            best_model_name,
            cv_summary,
        )
        for model_name in MODEL_ORDER
    ]

    task_output = {
        "task": task_name,
        "display_name": config["display_name"],
        "role": config["role"],
        "problem_type": config["problem_type"],
        "target": target_name,
        "row_count": int(frame.shape[0]),
        "feature_count": int(feature_frame.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "class_distribution": target_series_raw.value_counts().to_dict(),
        "partition_strategy": {
            "random_state": RANDOM_STATE,
            "split_sequence": "stratified_train_validation_test",
            "train_ratio": 0.60,
            "validation_ratio": 0.20,
            "test_ratio": 0.20,
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "cross_validation": {
                "applied_to_selected_model_only": True,
                "folds": CV_FOLDS,
                "selection_metric": primary_metric,
            },
            "leakage_controls": [
                "stratified splitting with a fixed random seed",
                "hold-out test set reserved until final evaluation",
                "all imputers, clipping bounds, encoders, and scaling fit inside the pipeline",
            ],
        },
        "outlier_summary": outlier_summary,
        "best_model": best_model_name,
        "model_selection_metric": primary_metric,
        "selected_model_cross_validation": cv_summary,
        "best_validation_metrics": {
            key: value
            for key, value in final_validation_metrics_by_model[best_model_name].items()
            if key != "classification_report"
        },
        "best_test_metrics": {
            key: value
            for key, value in final_test_metrics_by_model[best_model_name].items()
            if key != "classification_report"
        },
        "models": model_results,
        "summary_rows": summary_rows,
    }

    if task_name == "uci_multiclass":
        best_article_model_name = max(
            ARTICLE_MODEL_NAMES,
            key=lambda name: final_validation_metrics_by_model[name][primary_metric],
        )
        reproduced_accuracy = final_test_metrics_by_model[best_article_model_name]["accuracy"]
        task_output["article_alignment"] = {
            "reference": ARTICLE_BASELINE_REFERENCE,
            "implemented_article_models": list(ARTICLE_MODEL_NAMES),
            "best_reproduced_article_model": best_article_model_name,
            "best_reproduced_test_accuracy": reproduced_accuracy,
            "accuracy_gap_vs_reported": float(
                reproduced_accuracy - ARTICLE_BASELINE_REFERENCE["reported_accuracy"]
            ),
            "notes": (
                "The UCI multiclass task is the direct reproduction benchmark. "
                "The binary early-warning tasks are project extensions rather than one-to-one paper reproductions."
            ),
        }

    if label_encoder is not None:
        task_output["encoded_class_mapping"] = {
            str(index): label
            for index, label in enumerate(label_encoder.classes_.tolist())
        }

    return task_output


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
        "test_size": TEST_SIZE,
        "validation_share_of_trainval": VALIDATION_SHARE_OF_TRAINVAL,
        "cv_folds": CV_FOLDS,
        "article_reference": ARTICLE_BASELINE_REFERENCE,
        "tasks": task_outputs,
    }
    (RESULTS / "baseline_metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
