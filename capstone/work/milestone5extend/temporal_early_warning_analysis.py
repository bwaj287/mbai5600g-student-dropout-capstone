import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CAPSTONE = ROOT.parents[1]
RAW_OULAD = CAPSTONE / "work" / "milestone3" / "data" / "raw" / "oulad"
MILESTONE4_DATA = CAPSTONE / "work" / "milestone4" / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


# ---------------------------------------------------------
# Experiment Settings
# ---------------------------------------------------------
TIME_WINDOWS = [35, 60, 75]
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SHARE_OF_TRAINVAL = 0.25
CV_FOLDS = 5
FIXED_MILESTONE5_THRESHOLD = 0.60
BOOTSTRAP_SAMPLES = 1000

KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]
DROP_COLUMNS = ["final_result", "id_student", "is_attrition"]

# These are the selected OULAD settings from Milestone 5. They stay fixed for
# every time window so the amount of observed data is the main change.
XGBOOST_PARAMS = {
    "n_estimators": 320,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.80,
    "min_child_weight": 1,
    "reg_lambda": 1.0,
}

FEATURE_FAMILY_LABELS = {
    "early_academic_progress": "Early academic progress",
    "early_engagement": "Early engagement",
    "program_setup": "Program setup",
    "demographics": "Demographics",
    "prior_preparation": "Prior preparation",
    "other": "Other",
}


# ---------------------------------------------------------
# Small Preprocessing Helper
# ---------------------------------------------------------
class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
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
        return np.asarray(input_features, dtype=object)


def ensure_dirs():
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def read_oulad_tables():
    print("Loading OULAD source tables...")
    student_info = pd.read_csv(RAW_OULAD / "studentInfo.csv")
    student_registration = pd.read_csv(RAW_OULAD / "studentRegistration.csv")
    courses = pd.read_csv(RAW_OULAD / "courses.csv")
    assessments = pd.read_csv(RAW_OULAD / "assessments.csv")
    student_assessment = pd.read_csv(RAW_OULAD / "studentAssessment.csv")
    vle = pd.read_csv(RAW_OULAD / "vle.csv")

    student_vle_columns = [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ]
    student_vle_types = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int32",
    }
    student_vle_path = RAW_OULAD / "studentVle.csv"
    if student_vle_path.exists():
        student_vle = pd.read_csv(
            student_vle_path,
            usecols=student_vle_columns,
            dtype=student_vle_types,
        )
    else:
        archive_path = RAW_OULAD / "oulad.zip"
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open("studentVle.csv") as student_vle_file:
                student_vle = pd.read_csv(
                    student_vle_file,
                    usecols=student_vle_columns,
                    dtype=student_vle_types,
                )
    student_vle = student_vle.loc[
        student_vle["date"] <= max(TIME_WINDOWS)
    ].copy()

    # id_site is unique in OULAD, so one lookup is enough to attach activity type.
    vle_lookup = vle[["id_site", "activity_type"]].drop_duplicates()
    if vle_lookup["id_site"].duplicated().any():
        raise ValueError("OULAD id_site values are not unique.")
    student_vle = student_vle.merge(vle_lookup, on="id_site", how="left")

    assessment_base = student_assessment.merge(
        assessments,
        on="id_assessment",
        how="inner",
        suffixes=("", "_scheduled"),
    )

    return {
        "student_info": student_info,
        "student_registration": student_registration,
        "courses": courses,
        "assessments": assessments,
        "assessment_base": assessment_base,
        "student_vle": student_vle,
    }


def prepare_student_base(tables):
    student_info = tables["student_info"].copy()
    student_info["is_attrition"] = (
        student_info["final_result"] == "Withdrawn"
    ).astype(int)

    student_base = student_info.merge(
        tables["student_registration"].drop(columns=["date_unregistration"]),
        on=KEY_COLUMNS,
        how="left",
    )
    student_base = student_base.merge(
        tables["courses"],
        on=["code_module", "code_presentation"],
        how="left",
    )
    # Keep the original studentInfo row order so the random split reproduces
    # the Milestone 5 train, validation, and test membership.
    return student_base.reset_index(drop=True)


def make_assessment_features(tables, cutoff_days):
    assessments = tables["assessments"]
    early_assessments = assessments.loc[
        assessments["date"].fillna(np.inf) <= cutoff_days
    ].copy()

    assessment_base = tables["assessment_base"]
    assessment_base = assessment_base.loc[
        (assessment_base["date"].fillna(np.inf) <= cutoff_days)
        & (assessment_base["date_submitted"].fillna(np.inf) <= cutoff_days)
    ].copy()
    assessment_base["weighted_score"] = (
        assessment_base["score"].fillna(0.0)
        * assessment_base["weight"].fillna(0.0)
        / 100.0
    )
    assessment_base["submission_delay_days"] = (
        assessment_base["date_submitted"] - assessment_base["date"]
    )
    assessment_base["late_submission"] = (
        assessment_base["submission_delay_days"] > 0
    ).astype(int)

    assessment_core = (
        assessment_base.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            assessment_submission_count_early=("score", "size"),
            assessment_score_mean_early=("score", "mean"),
            assessment_score_std_early=("score", "std"),
            assessment_score_max_early=("score", "max"),
            assessment_score_min_early=("score", "min"),
            assessment_weighted_score_sum_early=("weighted_score", "sum"),
            assessment_mean_submission_delay_early=(
                "submission_delay_days",
                "mean",
            ),
            assessment_late_submission_count_early=("late_submission", "sum"),
            assessment_banked_count_early=("is_banked", "sum"),
        )
    )

    assessment_type = (
        assessment_base.pivot_table(
            index=KEY_COLUMNS,
            columns="assessment_type",
            values="score",
            aggfunc="size",
            fill_value=0,
        )
        .rename(
            columns=lambda value: (
                f"assessment_type_count_{str(value).lower()}_early"
            )
        )
        .reset_index()
    )

    assessment_schedule = (
        early_assessments.groupby(
            ["code_module", "code_presentation"],
            as_index=False,
        )
        .agg(
            early_assessment_count_expected=("id_assessment", "nunique"),
            early_assessment_weight_expected=("weight", "sum"),
        )
    )
    return assessment_schedule, assessment_core, assessment_type


def make_engagement_features(tables, cutoff_days):
    student_vle = tables["student_vle"]
    early_vle = student_vle.loc[student_vle["date"] <= cutoff_days].copy()

    vle_core = (
        early_vle.groupby(KEY_COLUMNS, as_index=False, observed=True)
        .agg(
            vle_total_clicks_early=("sum_click", "sum"),
            vle_event_count_early=("sum_click", "size"),
            vle_active_days_early=("date", "nunique"),
            vle_unique_sites_early=("id_site", "nunique"),
        )
    )

    vle_activity = (
        early_vle.pivot_table(
            index=KEY_COLUMNS,
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        .rename(
            columns=lambda value: f"vle_clicks_{str(value).lower()}_early"
        )
        .reset_index()
    )

    # Aggregation keeps category dtypes from the large click table. Converting
    # only the small result tables makes later merges match studentInfo.
    for frame in (vle_core, vle_activity):
        frame["code_module"] = frame["code_module"].astype(str)
        frame["code_presentation"] = frame["code_presentation"].astype(str)
    return vle_core, vle_activity


def build_window_dataset(student_base, tables, cutoff_days):
    print(f"Creating cumulative feature table through day {cutoff_days}...")
    assessment_schedule, assessment_core, assessment_type = (
        make_assessment_features(tables, cutoff_days)
    )
    vle_core, vle_activity = make_engagement_features(tables, cutoff_days)

    model_ready = student_base.merge(
        assessment_schedule,
        on=["code_module", "code_presentation"],
        how="left",
    )
    for feature_frame in (
        assessment_core,
        assessment_type,
        vle_core,
        vle_activity,
    ):
        model_ready = model_ready.merge(
            feature_frame,
            on=KEY_COLUMNS,
            how="left",
        )

    aggregate_columns = [
        column
        for column in model_ready.columns
        if column.endswith("_early")
        or column
        in {
            "early_assessment_count_expected",
            "early_assessment_weight_expected",
        }
    ]
    model_ready[aggregate_columns] = model_ready[aggregate_columns].fillna(0)
    model_ready["assessment_score_std_early"] = model_ready[
        "assessment_score_std_early"
    ].fillna(0)
    model_ready["assessment_submission_ratio_early"] = np.where(
        model_ready["early_assessment_count_expected"] > 0,
        model_ready["assessment_submission_count_early"]
        / model_ready["early_assessment_count_expected"],
        0.0,
    )
    model_ready["assessment_weighted_score_ratio_early"] = np.where(
        model_ready["early_assessment_weight_expected"] > 0,
        model_ready["assessment_weighted_score_sum_early"]
        / model_ready["early_assessment_weight_expected"],
        0.0,
    )
    model_ready["registration_lead_days"] = (
        -model_ready["date_registration"]
    ).clip(lower=0)
    return model_ready.reset_index(drop=True)


def use_common_feature_schema(window_frames):
    reference_path = (
        MILESTONE4_DATA / "oulad_binary_early_model_ready.csv"
    )
    reference_columns = pd.read_csv(reference_path, nrows=0).columns.tolist()

    extra_columns = sorted(
        {
            column
            for frame in window_frames.values()
            for column in frame.columns
            if column not in reference_columns
        }
    )
    common_columns = reference_columns + extra_columns

    for cutoff_days, frame in window_frames.items():
        frame = frame.reindex(columns=common_columns)
        time_feature_columns = [
            column
            for column in frame.columns
            if column.endswith("_early")
            or column
            in {
                "early_assessment_count_expected",
                "early_assessment_weight_expected",
            }
        ]
        frame[time_feature_columns] = frame[time_feature_columns].fillna(0)
        window_frames[cutoff_days] = frame
    return window_frames


def check_day75_reproduction(day75_frame):
    reference_path = (
        MILESTONE4_DATA / "oulad_binary_early_model_ready.csv"
    )
    reference = pd.read_csv(reference_path)
    candidate = day75_frame[reference.columns]

    row_keys_match = reference[KEY_COLUMNS].equals(candidate[KEY_COLUMNS])
    common_numeric = reference.select_dtypes(include=[np.number]).columns
    numeric_difference = (
        reference[common_numeric].fillna(0)
        - candidate[common_numeric].fillna(0)
    ).abs()
    maximum_numeric_difference = float(numeric_difference.to_numpy().max())

    categorical_columns = [
        column
        for column in reference.columns
        if column not in common_numeric
    ]
    categorical_match = (
        reference[categorical_columns].fillna("<missing>").astype(str)
        == candidate[categorical_columns].fillna("<missing>").astype(str)
    ).all().all()

    check = {
        "reference_rows": int(len(reference)),
        "candidate_rows": int(len(candidate)),
        "reference_columns": int(len(reference.columns)),
        "candidate_columns": int(len(candidate.columns)),
        "row_keys_match": bool(row_keys_match),
        "categorical_values_match": bool(categorical_match),
        "maximum_numeric_difference": maximum_numeric_difference,
        "exact_reproduction": bool(
            len(reference) == len(candidate)
            and row_keys_match
            and categorical_match
            and maximum_numeric_difference < 1e-10
        ),
    }
    (RESULTS / "day75_reproduction_check.json").write_text(
        json.dumps(check, indent=2)
    )
    if not check["exact_reproduction"]:
        raise ValueError(
            "The new day-75 feature table does not reproduce Milestone 4."
        )
    return check


def make_split_indices(target):
    all_indices = np.arange(len(target))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=VALIDATION_SHARE_OF_TRAINVAL,
        random_state=RANDOM_STATE,
        stratify=target.iloc[train_val_indices],
    )
    return {
        "train": train_indices,
        "validation": validation_indices,
        "train_validation": train_val_indices,
        "test": test_indices,
    }


def split_feature_types(frame):
    categorical_columns = frame.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = [
        column
        for column in frame.columns
        if column not in categorical_columns
    ]
    return categorical_columns, numeric_columns


def make_pipeline(features, target):
    categorical_columns, numeric_columns = split_feature_types(features)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("clipper", QuantileClipper()),
                    ]
                ),
                numeric_columns,
            ),
        ]
    )

    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    scale_pos_weight = (
        negative_count / positive_count if positive_count else 1.0
    )
    model = XGBClassifier(
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        objective="binary:logistic",
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        **XGBOOST_PARAMS,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )
    return pipeline


def tune_threshold(y_true, y_score):
    rows = []
    for threshold in np.round(np.arange(0.20, 0.81, 0.02), 2):
        predictions = (y_score >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(y_true, predictions, zero_division=0)
                ),
                "recall": float(
                    recall_score(y_true, predictions, zero_division=0)
                ),
                "f1": float(
                    f1_score(y_true, predictions, zero_division=0)
                ),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_true, predictions)
                ),
            }
        )
    threshold_frame = pd.DataFrame(rows)
    best_row = threshold_frame.sort_values(
        ["f1", "recall", "balanced_accuracy"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_frame


def calculate_metrics(y_true, y_score, threshold):
    predictions = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(
            average_precision_score(y_true, y_score)
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
        ).tolist(),
    }


def bootstrap_intervals(y_true, y_score, threshold, seed):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    values = {"f1": [], "roc_auc": [], "average_precision": []}

    for _ in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[sample]
        if np.unique(sample_y).size < 2:
            continue
        sample_score = y_score[sample]
        sample_prediction = (sample_score >= threshold).astype(int)
        values["f1"].append(
            f1_score(sample_y, sample_prediction, zero_division=0)
        )
        values["roc_auc"].append(roc_auc_score(sample_y, sample_score))
        values["average_precision"].append(
            average_precision_score(sample_y, sample_score)
        )

    intervals = {}
    for metric, metric_values in values.items():
        lower, upper = np.percentile(metric_values, [2.5, 97.5])
        intervals[metric] = {
            "lower": float(lower),
            "upper": float(upper),
        }
    return intervals


def cross_validation_results(features, target):
    pipeline = make_pipeline(features, target)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scores = cross_validate(
        pipeline,
        features,
        target,
        cv=cv,
        scoring={
            "f1": "f1",
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
        },
        n_jobs=None,
    )
    summary = {}
    for metric in ("f1", "roc_auc", "average_precision"):
        values = scores[f"test_{metric}"]
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return summary


def clean_feature_name(feature):
    for prefix in ("numeric__", "categorical__"):
        if feature.startswith(prefix):
            return feature.replace(prefix, "", 1)
    return feature


def classify_feature_family(feature):
    feature = clean_feature_name(feature)
    if feature.startswith("vle_"):
        return "early_engagement"
    if feature.startswith("assessment_") or feature.startswith(
        "early_assessment"
    ):
        return "early_academic_progress"
    if feature.startswith(
        (
            "code_module",
            "code_presentation",
            "studied_credits",
            "module_presentation_length",
            "date_registration",
            "registration_lead_days",
        )
    ):
        return "program_setup"
    if feature.startswith(
        ("gender", "region", "imd_band", "age_band", "disability")
    ):
        return "demographics"
    if feature.startswith(("highest_education", "num_of_prev_attempts")):
        return "prior_preparation"
    return "other"


def calculate_shap(pipeline, x_test, cutoff_days, max_rows=500):
    rng = np.random.default_rng(RANDOM_STATE)
    if len(x_test) > max_rows:
        sample_positions = rng.choice(
            len(x_test),
            size=max_rows,
            replace=False,
        )
        x_sample = x_test.iloc[sample_positions]
    else:
        x_sample = x_test

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed_sample = preprocessor.transform(x_sample)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_sample)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        mean_absolute_shap = np.abs(shap_array).mean(axis=(0, 2))
    else:
        mean_absolute_shap = np.abs(shap_array).mean(axis=0)

    shap_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_absolute_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_frame["feature_family"] = shap_frame["feature"].map(
        classify_feature_family
    )
    shap_frame.to_csv(
        RESULTS / f"day_{cutoff_days}_shap_summary.csv",
        index=False,
    )

    family_frame = (
        shap_frame.groupby("feature_family", as_index=False)[
            "mean_abs_shap"
        ]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    total_shap = family_frame["mean_abs_shap"].sum()
    family_frame["share_of_total_shap"] = (
        family_frame["mean_abs_shap"] / total_shap
    )
    family_frame["window_days"] = cutoff_days
    return shap_frame, family_frame


def save_shap_plot(shap_frame, cutoff_days):
    top_features = shap_frame.head(12).iloc[::-1].copy()
    top_features["feature"] = top_features["feature"].map(clean_feature_name)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(
        top_features["feature"],
        top_features["mean_abs_shap"],
        color="#356f8d",
    )
    ax.set_title(f"Top SHAP Features Through Day {cutoff_days}")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(
        FIGURES / f"day_{cutoff_days}_shap_top_features.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_confusion_matrices(model_outputs):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    labels = ["Not withdrawn", "Withdrawn"]

    for ax, cutoff_days in zip(axes, TIME_WINDOWS):
        matrix = np.asarray(
            model_outputs[cutoff_days]["fixed_metrics"]["confusion_matrix"]
        )
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"Day {cutoff_days}")
        ax.set_xticks([0, 1], labels=labels, rotation=25, ha="right")
        ax.set_yticks([0, 1], labels=labels)
        ax.set_xlabel("Predicted")
        if cutoff_days == TIME_WINDOWS[0]:
            ax.set_ylabel("Actual")
        for row_index in range(2):
            for column_index in range(2):
                ax.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                )

    fig.suptitle(
        "Held-out Test Confusion Matrices at Fixed Threshold 0.60",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "temporal_confusion_matrices.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_performance_plot(metrics_frame):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    windows = metrics_frame["window_days"]

    axes[0].plot(
        windows,
        metrics_frame["fixed_f1"],
        marker="o",
        linewidth=2,
        label="F1, fixed threshold 0.60",
    )
    axes[0].plot(
        windows,
        metrics_frame["tuned_f1"],
        marker="s",
        linewidth=2,
        label="F1, validation-tuned threshold",
    )
    axes[0].set_title("Withdrawal Classification F1")
    axes[0].set_xlabel("Cumulative observation window (days)")
    axes[0].set_ylabel("F1")
    axes[0].set_xticks(TIME_WINDOWS)
    axes[0].set_ylim(0.60, 0.82)
    axes[0].legend()

    axes[1].plot(
        windows,
        metrics_frame["roc_auc"],
        marker="o",
        linewidth=2,
        color="#b4583c",
        label="ROC-AUC",
    )
    axes[1].plot(
        windows,
        metrics_frame["average_precision"],
        marker="s",
        linewidth=2,
        color="#55843c",
        label="Average precision",
    )
    axes[1].set_title("Threshold-free Test Performance")
    axes[1].set_xlabel("Cumulative observation window (days)")
    axes[1].set_ylabel("Score")
    axes[1].set_xticks(TIME_WINDOWS)
    axes[1].set_ylim(0.70, 0.95)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        FIGURES / "temporal_performance_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_family_plot(family_frame):
    plot_frame = family_frame.copy()
    plot_frame["feature_family"] = plot_frame["feature_family"].map(
        FEATURE_FAMILY_LABELS
    )
    pivot = plot_frame.pivot_table(
        index="feature_family",
        columns="window_days",
        values="share_of_total_shap",
        fill_value=0,
    )
    pivot = pivot.reindex(columns=TIME_WINDOWS).fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    pivot.plot(kind="barh", ax=ax)
    ax.set_title("How SHAP Feature-Family Contributions Change Over Time")
    ax.set_xlabel("Share of total mean absolute SHAP")
    ax.set_ylabel("")
    ax.legend(
        [f"Day {window}" for window in TIME_WINDOWS],
        title="Window",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "temporal_shap_family_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def paired_bootstrap_differences(predictions_by_window):
    comparisons = [(35, 60), (60, 75), (35, 75)]
    y_true = predictions_by_window[TIME_WINDOWS[0]]["y_true"]
    rng = np.random.default_rng(RANDOM_STATE + 500)
    rows = []

    for earlier_window, later_window in comparisons:
        earlier_scores = predictions_by_window[earlier_window]["scores"]
        later_scores = predictions_by_window[later_window]["scores"]
        f1_differences = []
        auc_differences = []

        for _ in range(BOOTSTRAP_SAMPLES):
            sample = rng.integers(0, len(y_true), len(y_true))
            sample_y = y_true[sample]
            if np.unique(sample_y).size < 2:
                continue
            earlier_sample = earlier_scores[sample]
            later_sample = later_scores[sample]
            earlier_prediction = (
                earlier_sample >= FIXED_MILESTONE5_THRESHOLD
            ).astype(int)
            later_prediction = (
                later_sample >= FIXED_MILESTONE5_THRESHOLD
            ).astype(int)
            f1_differences.append(
                f1_score(sample_y, later_prediction, zero_division=0)
                - f1_score(sample_y, earlier_prediction, zero_division=0)
            )
            auc_differences.append(
                roc_auc_score(sample_y, later_sample)
                - roc_auc_score(sample_y, earlier_sample)
            )

        for metric, differences in (
            ("f1", f1_differences),
            ("roc_auc", auc_differences),
        ):
            lower, upper = np.percentile(differences, [2.5, 97.5])
            rows.append(
                {
                    "comparison": (
                        f"day_{later_window}_minus_day_{earlier_window}"
                    ),
                    "metric": metric,
                    "mean_difference": float(np.mean(differences)),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "ci_excludes_zero": bool(lower > 0 or upper < 0),
                }
            )

    difference_frame = pd.DataFrame(rows)
    difference_frame.to_csv(
        RESULTS / "paired_window_differences.csv",
        index=False,
    )
    return difference_frame


def run_one_window(frame, cutoff_days, split_indices):
    target = frame["is_attrition"].astype(int)
    features = frame.drop(
        columns=[column for column in DROP_COLUMNS if column in frame.columns]
    )

    train_index = split_indices["train"]
    validation_index = split_indices["validation"]
    train_validation_index = split_indices["train_validation"]
    test_index = split_indices["test"]

    x_train = features.iloc[train_index]
    y_train = target.iloc[train_index]
    x_validation = features.iloc[validation_index]
    y_validation = target.iloc[validation_index]
    x_train_validation = features.iloc[train_validation_index]
    y_train_validation = target.iloc[train_validation_index]
    x_test = features.iloc[test_index]
    y_test = target.iloc[test_index]

    print(f"Training day-{cutoff_days} validation model...")
    validation_pipeline = make_pipeline(x_train, y_train)
    validation_pipeline.fit(x_train, y_train)
    validation_scores = validation_pipeline.predict_proba(x_validation)[:, 1]
    selected_threshold, threshold_frame = tune_threshold(
        y_validation,
        validation_scores,
    )
    threshold_frame["window_days"] = cutoff_days

    print(f"Training day-{cutoff_days} final model...")
    final_pipeline = make_pipeline(
        x_train_validation,
        y_train_validation,
    )
    final_pipeline.fit(x_train_validation, y_train_validation)
    test_scores = final_pipeline.predict_proba(x_test)[:, 1]

    fixed_metrics = calculate_metrics(
        y_test,
        test_scores,
        FIXED_MILESTONE5_THRESHOLD,
    )
    tuned_metrics = calculate_metrics(
        y_test,
        test_scores,
        selected_threshold,
    )
    confidence_intervals = bootstrap_intervals(
        y_test,
        test_scores,
        FIXED_MILESTONE5_THRESHOLD,
        RANDOM_STATE + cutoff_days,
    )

    print(f"Running day-{cutoff_days} five-fold cross-validation...")
    cv_results = cross_validation_results(
        x_train_validation,
        y_train_validation,
    )

    print(f"Calculating day-{cutoff_days} SHAP values...")
    shap_frame, family_frame = calculate_shap(
        final_pipeline,
        x_test,
        cutoff_days,
    )
    save_shap_plot(shap_frame, cutoff_days)

    output = {
        "window_days": cutoff_days,
        "row_count": int(len(frame)),
        "feature_count_before_encoding": int(features.shape[1]),
        "train_rows": int(len(train_index)),
        "validation_rows": int(len(validation_index)),
        "test_rows": int(len(test_index)),
        "selected_validation_threshold": selected_threshold,
        "fixed_threshold": FIXED_MILESTONE5_THRESHOLD,
        "fixed_metrics": fixed_metrics,
        "tuned_metrics": tuned_metrics,
        "fixed_threshold_bootstrap_95_ci": confidence_intervals,
        "cross_validation": cv_results,
        "top_shap_features": [
            {
                "feature": clean_feature_name(row["feature"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
            }
            for _, row in shap_frame.head(10).iterrows()
        ],
    }
    predictions = {
        "y_true": y_test.to_numpy(),
        "scores": test_scores,
    }
    return output, threshold_frame, family_frame, predictions


def make_metrics_frame(model_outputs):
    rows = []
    for cutoff_days in TIME_WINDOWS:
        output = model_outputs[cutoff_days]
        fixed = output["fixed_metrics"]
        tuned = output["tuned_metrics"]
        intervals = output["fixed_threshold_bootstrap_95_ci"]
        cv = output["cross_validation"]
        rows.append(
            {
                "window_days": cutoff_days,
                "model": "XGBoost",
                "features_before_encoding": output[
                    "feature_count_before_encoding"
                ],
                "fixed_threshold": output["fixed_threshold"],
                "validation_selected_threshold": output[
                    "selected_validation_threshold"
                ],
                "fixed_accuracy": fixed["accuracy"],
                "fixed_balanced_accuracy": fixed["balanced_accuracy"],
                "fixed_precision": fixed["precision"],
                "fixed_recall": fixed["recall"],
                "fixed_f1": fixed["f1"],
                "fixed_f1_ci_95_lower": intervals["f1"]["lower"],
                "fixed_f1_ci_95_upper": intervals["f1"]["upper"],
                "tuned_precision": tuned["precision"],
                "tuned_recall": tuned["recall"],
                "tuned_f1": tuned["f1"],
                "roc_auc": fixed["roc_auc"],
                "roc_auc_ci_95_lower": intervals["roc_auc"]["lower"],
                "roc_auc_ci_95_upper": intervals["roc_auc"]["upper"],
                "average_precision": fixed["average_precision"],
                "average_precision_ci_95_lower": intervals[
                    "average_precision"
                ]["lower"],
                "average_precision_ci_95_upper": intervals[
                    "average_precision"
                ]["upper"],
                "cv_f1_mean": cv["f1"]["mean"],
                "cv_f1_std": cv["f1"]["std"],
                "cv_roc_auc_mean": cv["roc_auc"]["mean"],
                "cv_roc_auc_std": cv["roc_auc"]["std"],
            }
        )
    return pd.DataFrame(rows)


def main():
    ensure_dirs()
    tables = read_oulad_tables()
    student_base = prepare_student_base(tables)

    window_frames = {}
    for cutoff_days in TIME_WINDOWS:
        window_frames[cutoff_days] = build_window_dataset(
            student_base,
            tables,
            cutoff_days,
        )
    window_frames = use_common_feature_schema(window_frames)
    reproduction_check = check_day75_reproduction(window_frames[75])

    target = window_frames[TIME_WINDOWS[0]]["is_attrition"].astype(int)
    split_indices = make_split_indices(target)

    first_keys = window_frames[TIME_WINDOWS[0]][KEY_COLUMNS]
    for cutoff_days in TIME_WINDOWS[1:]:
        if not first_keys.equals(window_frames[cutoff_days][KEY_COLUMNS]):
            raise ValueError("Student rows are not aligned across time windows.")
        if not target.equals(
            window_frames[cutoff_days]["is_attrition"].astype(int)
        ):
            raise ValueError("Target labels changed across time windows.")

    model_outputs = {}
    threshold_frames = []
    family_frames = []
    predictions_by_window = {}

    for cutoff_days in TIME_WINDOWS:
        output, threshold_frame, family_frame, predictions = run_one_window(
            window_frames[cutoff_days],
            cutoff_days,
            split_indices,
        )
        model_outputs[cutoff_days] = output
        threshold_frames.append(threshold_frame)
        family_frames.append(family_frame)
        predictions_by_window[cutoff_days] = predictions

    metrics_frame = make_metrics_frame(model_outputs)
    metrics_frame.to_csv(
        RESULTS / "temporal_model_metrics.csv",
        index=False,
    )
    pd.concat(threshold_frames, ignore_index=True).to_csv(
        RESULTS / "threshold_tuning_by_window.csv",
        index=False,
    )
    family_frame = pd.concat(family_frames, ignore_index=True)
    family_frame.to_csv(
        RESULTS / "shap_family_by_window.csv",
        index=False,
    )
    paired_differences = paired_bootstrap_differences(
        predictions_by_window
    )

    save_performance_plot(metrics_frame)
    save_confusion_matrices(model_outputs)
    save_family_plot(family_frame)

    payload = {
        "experiment": {
            "time_windows": TIME_WINDOWS,
            "random_state": RANDOM_STATE,
            "split": "60% train, 20% validation, 20% held-out test",
            "same_student_split_across_windows": True,
            "fixed_milestone5_threshold": FIXED_MILESTONE5_THRESHOLD,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "cv_folds": CV_FOLDS,
            "xgboost_params": XGBOOST_PARAMS,
        },
        "day75_reproduction_check": reproduction_check,
        "windows": {
            str(cutoff_days): model_outputs[cutoff_days]
            for cutoff_days in TIME_WINDOWS
        },
        "paired_bootstrap_differences": paired_differences.to_dict(
            orient="records"
        ),
    }
    (RESULTS / "temporal_model_metrics.json").write_text(
        json.dumps(payload, indent=2)
    )

    print("\nTemporal comparison")
    print(
        metrics_frame[
            [
                "window_days",
                "fixed_f1",
                "tuned_f1",
                "roc_auc",
                "average_precision",
                "validation_selected_threshold",
            ]
        ].round(4)
    )
    print("\nSaved results to:", RESULTS)
    print("Saved figures to:", FIGURES)


if __name__ == "__main__":
    main()
