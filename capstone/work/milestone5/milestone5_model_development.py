import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

import shap


# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CAPSTONE = ROOT.parents[1]
REPO_ROOT = CAPSTONE.parent
MILESTONE4 = ROOT.parent / "milestone4"
M4_DATA = MILESTONE4 / "data"
M4_RESULTS = MILESTONE4 / "results"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"

# ---------------------------------------------------------
# Project Settings
# ---------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SHARE_OF_TRAINVAL = 0.25
CV_FOLDS = 5
OUTLIER_CLIP_LOWER = 0.01
OUTLIER_CLIP_UPPER = 0.99

# UCI stores many categorical fields as numbers. Keeping this list explicit
# avoids accidentally treating every coded field as a continuous measurement.
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

# ---------------------------------------------------------
# Modeling Tasks
# ---------------------------------------------------------
TASKS = {
    "uci_multiclass": {
        "display_name": "UCI multiclass reproduction",
        "path": M4_DATA / "uci_multiclass_model_ready.csv",
        "target": "Target",
        "drop_columns": ["is_attrition"],
        "problem_type": "multiclass",
        "primary_metric": "macro_f1",
        "positive_label": None,
    },
    "uci_binary_early": {
        "display_name": "UCI binary early warning",
        "path": M4_DATA / "uci_binary_early_model_ready.csv",
        "target": "is_attrition",
        # Milestone 4 already removed the second-semester fields for leakage control.
        "drop_columns": ["Target"],
        "problem_type": "binary",
        "primary_metric": "f1",
        "positive_label": "Dropout",
    },
    "oulad_binary_early": {
        "display_name": "OULAD binary early warning",
        "path": M4_DATA / "oulad_binary_early_model_ready.csv",
        "target": "is_attrition",
        # id_student is only an identifier, and final_result is the source of the label.
        "drop_columns": ["final_result", "id_student"],
        "problem_type": "binary",
        "primary_metric": "f1",
        "positive_label": "Withdrawn",
    },
}

# ---------------------------------------------------------
# Hyperparameter Grids
# ---------------------------------------------------------
# The grids are intentionally focused. This keeps the run practical on a laptop
# while still testing the parameters that matter most for this milestone.
MODEL_SEARCHES = {
    "uci_multiclass": {
        "xgboost": {
            "n_estimators": [180, 260],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.07],
            "subsample": [0.85],
            "colsample_bytree": [0.8],
            "min_child_weight": [1, 3],
            "reg_lambda": [1.0],
        },
    },
    "uci_binary_early": {
        "logistic_regression": {
            "C": [0.05, 0.1, 0.5, 1.0, 2.0],
            "class_weight": ["balanced"],
        },
        "xgboost": {
            "n_estimators": [180, 260],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.07],
            "subsample": [0.85],
            "colsample_bytree": [0.8],
            "min_child_weight": [1, 3],
            "reg_lambda": [1.0],
            "scale_pos_weight": ["auto"],
        },
    },
    "oulad_binary_early": {
        "xgboost": {
            "n_estimators": [220, 320],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.85],
            "colsample_bytree": [0.8],
            "min_child_weight": [1, 5],
            "reg_lambda": [1.0],
            "scale_pos_weight": ["auto"],
        },
    },
}


# ---------------------------------------------------------
# Small Preprocessing Helper
# ---------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=OUTLIER_CLIP_LOWER, upper_quantile=OUTLIER_CLIP_UPPER):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x, y=None):
        # Clip instead of deleting rows because some extreme learning behavior is real.
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


def ensure_dirs():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def rel(path):
    return str(path.relative_to(REPO_ROOT))


def split_feature_types(frame):
    categorical_columns = frame.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_columns = [column for column in frame.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def prepare_task_data(task_name, config):
    # Load the model-ready table created in Milestone 4.
    frame = pd.read_csv(config["path"])
    print("\nLoaded:", config["display_name"])
    print("Rows and columns:", frame.shape)

    target_name = config["target"]
    drop_columns = set(config["drop_columns"]) | {target_name}
    feature_frame = frame.drop(columns=[column for column in drop_columns if column in frame.columns])

    if task_name.startswith("uci_"):
        uci_categorical_columns = [
            column for column in feature_frame.columns if column not in UCI_CONTINUOUS_COLUMNS
        ]
        for column in uci_categorical_columns:
            feature_frame[column] = feature_frame[column].astype("string")

    target_raw = frame[target_name]
    label_encoder = None
    if config["problem_type"] == "multiclass":
        label_encoder = LabelEncoder()
        target = pd.Series(label_encoder.fit_transform(target_raw), index=target_raw.index, name=target_name)
        class_names = label_encoder.classes_.tolist()
    else:
        target = target_raw.astype(int)
        class_names = ["not_attrition", "attrition"]

    print("Target distribution:")
    print(target_raw.value_counts())

    return frame, feature_frame, target, target_raw, label_encoder, class_names


def build_preprocessor(categorical_columns, numeric_columns, model_name):
    # Same idea as Milestone 4: impute first, then clip numeric outliers.
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper()),
    ]
    if model_name == "logistic_regression":
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
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


def make_estimator(
    model_name,
    problem_type,
    params,
    y_train,
    class_count,
):
    params = dict(params)
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
            **params,
        )

    if model_name == "xgboost":
        # Let XGBoost account for the smaller attrition class instead of manually
        # over-sampling students. That keeps the comparison closer to Milestone 4.
        if params.get("scale_pos_weight") == "auto":
            positive_count = int((y_train == 1).sum())
            negative_count = int((y_train == 0).sum())
            params["scale_pos_weight"] = negative_count / positive_count if positive_count else 1.0

        base_params = {
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "eval_metric": "mlogloss" if problem_type == "multiclass" else "logloss",
            "objective": "multi:softprob" if problem_type == "multiclass" else "binary:logistic",
            "verbosity": 0,
        }
        if problem_type == "multiclass":
            base_params["num_class"] = class_count
        return XGBClassifier(**base_params, **params)

    raise ValueError(f"Unsupported tuned model: {model_name}")


def build_pipeline(
    model_name,
    problem_type,
    params,
    categorical_columns,
    numeric_columns,
    y_train,
    class_count,
):
    preprocessor = build_preprocessor(categorical_columns, numeric_columns, model_name)
    estimator = make_estimator(model_name, problem_type, params, y_train, class_count)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def get_score(problem_type, pipeline, features):
    probabilities = pipeline.predict_proba(features)
    if problem_type == "binary":
        return probabilities[:, 1]
    return probabilities


def metrics_for_predictions(
    problem_type,
    y_true,
    y_pred,
    y_score,
):
    if problem_type == "multiclass":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    return metrics


def tune_threshold(y_true, y_score):
    rows = []
    # Early-warning systems do not have to use the default 0.50 cutoff. We choose
    # the threshold on validation data so the test set stays untouched.
    for threshold in np.round(np.arange(0.20, 0.81, 0.02), 2):
        y_pred = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            }
        )
    threshold_frame = pd.DataFrame(rows)
    best_row = threshold_frame.sort_values(
        ["f1", "recall", "balanced_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_frame


def save_threshold_plot(task_name, model_name, threshold_frame, selected_threshold):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(threshold_frame["threshold"], threshold_frame["precision"], label="precision")
    ax.plot(threshold_frame["threshold"], threshold_frame["recall"], label="recall")
    ax.plot(threshold_frame["threshold"], threshold_frame["f1"], label="f1")
    ax.axvline(selected_threshold, color="black", linestyle="--", label=f"selected={selected_threshold:.2f}")
    ax.set_title(f"Threshold Tuning: {task_name} - {model_name}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path = FIGURES / f"{task_name}_{model_name}_threshold_curve.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(out_path)


def save_confusion_matrix(
    task_name,
    model_name,
    y_true,
    y_pred,
    class_names,
):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Tuned Model: {task_name} - {model_name}")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(column_index, row_index, str(matrix[row_index, column_index]), ha="center", va="center")
    fig.tight_layout()
    out_path = FIGURES / f"{task_name}_{model_name}_tuned_confusion_matrix.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(out_path)


def extract_feature_importance(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    if isinstance(classifier, LogisticRegression):
        coefficients = classifier.coef_
        if coefficients.ndim == 1:
            importances = np.abs(coefficients)
        else:
            importances = np.abs(coefficients).mean(axis=0)
    else:
        importances = classifier.feature_importances_

    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )


def save_feature_importance(task_name, model_name, importance_frame):
    csv_path = RESULTS / f"{task_name}_{model_name}_tuned_feature_importance.csv"
    importance_frame.to_csv(csv_path, index=False)

    top_features = importance_frame.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f8f")
    ax.set_title(f"Tuned Top Features: {task_name} - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    plot_path = FIGURES / f"{task_name}_{model_name}_tuned_top_features.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(csv_path), rel(plot_path)


def save_shap_summary(
    task_name,
    model_name,
    fitted_pipeline,
    x_test,
    max_rows=500,
):
    if model_name != "xgboost":
        return None

    # SHAP can be slow on the OULAD test set, so we explain a reproducible sample.
    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_test) > max_rows:
        sample_indices = rng.choice(len(x_test), size=max_rows, replace=False)
        x_sample = x_test.iloc[sample_indices]
    else:
        x_sample = x_test

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]
    transformed_sample = preprocessor.transform(x_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_sample)
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(values).mean(axis=0) for values in shap_values], axis=0)
    else:
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            mean_abs = np.abs(shap_array).mean(axis=(0, 2))
        else:
            mean_abs = np.abs(shap_array).mean(axis=0)

    shap_frame = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    csv_path = RESULTS / f"{task_name}_{model_name}_shap_summary.csv"
    shap_frame.to_csv(csv_path, index=False)

    top_features = shap_frame.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["mean_abs_shap"], color="#7a4f9a")
    ax.set_title(f"Mean Absolute SHAP: {task_name} - {model_name}")
    ax.set_xlabel("Mean absolute SHAP value")
    fig.tight_layout()
    plot_path = FIGURES / f"{task_name}_{model_name}_shap_summary.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "shap_summary_path": rel(csv_path),
        "shap_plot_path": rel(plot_path),
        "shap_rows_sampled": int(len(x_sample)),
    }


def read_baseline_rows():
    baseline_path = M4_RESULTS / "baseline_comparison.csv"
    if not baseline_path.exists():
        return pd.DataFrame()
    return pd.read_csv(baseline_path)


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [clean_for_json(inner_value) for inner_value in value]
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def baseline_reference_for_task(baseline_frame, task_name, primary_metric):
    if baseline_frame.empty:
        return {}
    task_rows = baseline_frame[baseline_frame["task"] == task_name].copy()
    selected_rows = task_rows[task_rows["selected_by_validation"] == True]
    if selected_rows.empty:
        selected_rows = task_rows.sort_values(f"validation_{primary_metric}", ascending=False).head(1)
    if selected_rows.empty:
        return {}
    row = selected_rows.iloc[0].to_dict()
    return {
        "baseline_model": row.get("model"),
        "baseline_test_primary_metric": row.get(f"test_{primary_metric}"),
        "baseline_test_accuracy": row.get("test_accuracy"),
        "baseline_test_roc_auc": row.get("test_roc_auc"),
    }


def build_cv_scoring(problem_type):
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


def cross_validation_summary(
    task_name,
    model_name,
    params,
    problem_type,
    categorical_columns,
    numeric_columns,
    x_train_val,
    y_train_val,
    class_count,
):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(
        model_name,
        problem_type,
        params,
        categorical_columns,
        numeric_columns,
        y_train_val,
        class_count,
    )
    cv_scores = cross_validate(
        pipeline,
        x_train_val,
        y_train_val,
        cv=cv,
        scoring=build_cv_scoring(problem_type),
        n_jobs=None,
    )
    summary = {}
    for key, values in cv_scores.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return {
        "task": task_name,
        "model": model_name,
        "folds": CV_FOLDS,
        "metrics": summary,
    }


def run_task(task_name, config, baseline_frame):
    frame, features, target, target_raw, label_encoder, class_names = prepare_task_data(task_name, config)
    categorical_columns, numeric_columns = split_feature_types(features)
    class_count = int(target.nunique())

    # 60/20/20 split: train candidates, choose settings on validation, report test once.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=VALIDATION_SHARE_OF_TRAINVAL,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    primary_metric = config["primary_metric"]
    baseline_reference = baseline_reference_for_task(baseline_frame, task_name, primary_metric)
    task_rows = []
    model_details = {}

    for model_name, grid in MODEL_SEARCHES[task_name].items():
        print("\nModel:", model_name)
        best_candidate = None
        candidate_count = 0

        for candidate_number, params in enumerate(ParameterGrid(grid), start=1):
            candidate_count += 1
            pipeline = build_pipeline(
                model_name,
                config["problem_type"],
                params,
                categorical_columns,
                numeric_columns,
                y_train,
                class_count,
            )
            pipeline.fit(x_train, y_train)
            validation_scores = get_score(config["problem_type"], pipeline, x_val)

            if config["problem_type"] == "binary":
                selected_threshold, threshold_frame = tune_threshold(y_val, validation_scores)
                validation_predictions = (validation_scores >= selected_threshold).astype(int)
            else:
                selected_threshold = None
                threshold_frame = None
                validation_predictions = pipeline.predict(x_val)

            validation_metrics = metrics_for_predictions(
                config["problem_type"],
                y_val,
                validation_predictions,
                validation_scores,
            )
            score = validation_metrics[primary_metric]
            candidate = {
                "candidate_number": candidate_number,
                "params": params,
                "selected_threshold": selected_threshold,
                "validation_metrics": validation_metrics,
                "threshold_frame": threshold_frame,
                "score": score,
            }
            if best_candidate is None or score > best_candidate["score"]:
                best_candidate = candidate

        print("Best validation", primary_metric, "=", round(best_candidate["score"], 4))
        best_params = best_candidate["params"]
        final_pipeline = build_pipeline(
            model_name,
            config["problem_type"],
            best_params,
            categorical_columns,
            numeric_columns,
            y_train_val,
            class_count,
        )
        final_pipeline.fit(x_train_val, y_train_val)
        test_scores = get_score(config["problem_type"], final_pipeline, x_test)
        if config["problem_type"] == "binary":
            selected_threshold = best_candidate["selected_threshold"]
            test_predictions = (test_scores >= selected_threshold).astype(int)
            threshold_path = save_threshold_plot(
                task_name,
                model_name,
                best_candidate["threshold_frame"],
                selected_threshold,
            )
        else:
            selected_threshold = None
            test_predictions = final_pipeline.predict(x_test)
            threshold_path = None

        test_metrics = metrics_for_predictions(
            config["problem_type"],
            y_test,
            test_predictions,
            test_scores,
        )
        confusion_path = save_confusion_matrix(task_name, model_name, y_test, test_predictions, class_names)
        importance_frame = extract_feature_importance(final_pipeline)
        importance_path, importance_plot_path = save_feature_importance(task_name, model_name, importance_frame)
        shap_outputs = save_shap_summary(task_name, model_name, final_pipeline, x_test)
        cv_summary = cross_validation_summary(
            task_name,
            model_name,
            best_params,
            config["problem_type"],
            categorical_columns,
            numeric_columns,
            x_train_val,
            y_train_val,
            class_count,
        )

        baseline_primary = baseline_reference.get("baseline_test_primary_metric")
        tuned_primary = test_metrics[primary_metric]
        task_row = {
            "task": task_name,
            "display_name": config["display_name"],
            "model": model_name,
            "primary_metric": primary_metric,
            "selected_threshold": selected_threshold,
            "validation_primary_metric": best_candidate["score"],
            "test_primary_metric": tuned_primary,
            "test_accuracy": test_metrics.get("accuracy"),
            "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
            "test_precision": test_metrics.get("precision"),
            "test_recall": test_metrics.get("recall"),
            "test_f1": test_metrics.get("f1"),
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_roc_auc": test_metrics.get("roc_auc"),
            "baseline_model": baseline_reference.get("baseline_model"),
            "baseline_test_primary_metric": baseline_primary,
            "test_primary_metric_change_vs_baseline": (
                float(tuned_primary - baseline_primary)
                if baseline_primary is not None and not pd.isna(baseline_primary)
                else None
            ),
            "candidate_count": candidate_count,
            "best_params": json.dumps(best_params, sort_keys=True),
            "confusion_matrix_path": confusion_path,
            "feature_importance_path": importance_path,
            "feature_plot_path": importance_plot_path,
            "threshold_plot_path": threshold_path,
        }
        if shap_outputs:
            task_row.update(shap_outputs)
        task_rows.append(task_row)

        print("Test", primary_metric, "=", round(tuned_primary, 4))
        if selected_threshold is not None:
            print("Selected threshold =", round(selected_threshold, 2))

        model_details[model_name] = {
            "best_params": best_params,
            "selected_threshold": selected_threshold,
            "validation_metrics": best_candidate["validation_metrics"],
            "test_metrics": test_metrics,
            "baseline_reference": baseline_reference,
            "candidate_count": candidate_count,
            "cross_validation": cv_summary,
            "paths": {
                "confusion_matrix": confusion_path,
                "feature_importance": importance_path,
                "feature_plot": importance_plot_path,
                "threshold_plot": threshold_path,
                **(shap_outputs or {}),
            },
        }

    task_payload = {
        "task": task_name,
        "display_name": config["display_name"],
        "problem_type": config["problem_type"],
        "target": config["target"],
        "row_count": int(frame.shape[0]),
        "feature_count": int(features.shape[1]),
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "class_distribution": target_raw.value_counts().to_dict(),
        "split": {
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_val)),
            "test_rows": int(len(x_test)),
            "random_state": RANDOM_STATE,
        },
        "baseline_reference": baseline_reference,
        "models": model_details,
    }
    if label_encoder is not None:
        task_payload["encoded_class_mapping"] = {
            str(index): label for index, label in enumerate(label_encoder.classes_.tolist())
        }
    return task_rows, task_payload


def main():
    ensure_dirs()
    baseline_frame = read_baseline_rows()
    all_rows = []
    task_payloads = {}

    print("=" * 70)
    print("Milestone 5: Advanced Modeling and Optimization")
    print("=" * 70)

    for task_name, config in TASKS.items():
        print("\n" + "-" * 70)
        print("Task:", task_name)
        print("-" * 70)
        task_rows, task_payload = run_task(task_name, config, baseline_frame)
        all_rows.extend(task_rows)
        task_payloads[task_name] = task_payload

    comparison_frame = pd.DataFrame(all_rows)
    comparison_path = RESULTS / "tuned_model_comparison.csv"
    comparison_frame.to_csv(comparison_path, index=False)

    payload = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_share_of_trainval": VALIDATION_SHARE_OF_TRAINVAL,
        "cv_folds": CV_FOLDS,
        "shap_available": True,
        "tasks": task_payloads,
    }
    metrics_path = RESULTS / "tuned_model_metrics.json"
    metrics_path.write_text(json.dumps(clean_for_json(payload), indent=2))

    print("\nSaved Milestone 5 comparison to", comparison_path)
    print("Saved Milestone 5 metrics to", metrics_path)
    print("\nFinal comparison table:")
    print(comparison_frame.to_string(index=False))


if __name__ == "__main__":
    main()
