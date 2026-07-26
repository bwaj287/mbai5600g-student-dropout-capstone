from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
CAPSTONE = ROOT.parents[1]
TEMPORAL_DIR = CAPSTONE / "work" / "milestone5extend"
sys.path.insert(0, str(TEMPORAL_DIR))

import temporal_early_warning_analysis as temporal


RESULTS = ROOT / "journal_results"
FIGURES = ROOT / "journal_figures"
TIME_WINDOWS = [35, 60, 75]
RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 1000
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]

NON_FEATURE_COLUMNS = [
    "final_result",
    "is_attrition",
    "future_withdrawal",
    "date_unregistration",
    "id_student",
]

XGBOOST_CANDIDATES = [
    {
        "n_estimators": 240,
        "max_depth": 4,
        "learning_rate": 0.05,
    },
    {
        "n_estimators": 320,
        "max_depth": 5,
        "learning_rate": 0.03,
    },
    {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.03,
    },
]

LOGISTIC_CANDIDATES = [0.05, 0.20, 1.00]

UCI_CONTINUOUS_COLUMNS = {
    "Application order",
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
}


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        self.lower_bounds_ = np.nanquantile(
            values,
            self.lower_quantile,
            axis=0,
        )
        self.upper_bounds_ = np.nanquantile(
            values,
            self.upper_quantile,
            axis=0,
        )
        return self

    def transform(self, x):
        values = np.asarray(x, dtype=float)
        return np.clip(
            values,
            self.lower_bounds_,
            self.upper_bounds_,
        )

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def ensure_dirs():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def build_student_base(tables):
    student_info = tables["student_info"].copy()
    registration = tables["student_registration"].copy()

    base = student_info.merge(
        registration,
        on=KEY_COLUMNS,
        how="left",
    )
    base = base.merge(
        tables["courses"],
        on=["code_module", "code_presentation"],
        how="left",
    )
    base["is_attrition"] = (
        base["final_result"] == "Withdrawn"
    ).astype(int)
    return base.reset_index(drop=True)


def build_window_frames(student_base, tables):
    frames = {}
    for cutoff in TIME_WINDOWS:
        frame = temporal.build_window_dataset(
            student_base,
            tables,
            cutoff,
        )
        frames[cutoff] = frame

    all_columns = sorted(
        set().union(*(set(frame.columns) for frame in frames.values()))
    )
    for cutoff in TIME_WINDOWS:
        frame = frames[cutoff].reindex(columns=all_columns)
        aggregate_columns = [
            column
            for column in frame.columns
            if column.endswith("_early")
            or column
            in {
                "early_assessment_count_expected",
                "early_assessment_weight_expected",
            }
        ]
        frame[aggregate_columns] = frame[aggregate_columns].fillna(0)
        frames[cutoff] = frame
    return frames


def make_landmark_cohort(frame, cutoff):
    withdrawal_timing_known = ~(
        frame["final_result"].eq("Withdrawn")
        & frame["date_unregistration"].isna()
    )
    registered_by_cutoff = (
        frame["date_registration"].notna()
        & frame["date_registration"].le(cutoff)
    )
    still_registered = (
        frame["date_unregistration"].isna()
        | frame["date_unregistration"].gt(cutoff)
    )

    cohort = frame.loc[
        withdrawal_timing_known
        & registered_by_cutoff
        & still_registered
    ].copy()
    cohort["future_withdrawal"] = (
        cohort["final_result"].eq("Withdrawn")
        & cohort["date_unregistration"].gt(cutoff)
    ).astype(int)
    return cohort.reset_index(drop=True)


def make_controlled_cohorts(window_frames, day75_cohort):
    keys_and_target = day75_cohort[
        KEY_COLUMNS + ["future_withdrawal"]
    ].copy()
    controlled = {}

    for cutoff in TIME_WINDOWS:
        frame = window_frames[cutoff].drop(
            columns=["future_withdrawal"],
            errors="ignore",
        )
        cohort = keys_and_target.merge(
            frame,
            on=KEY_COLUMNS,
            how="left",
        )
        controlled[cutoff] = cohort
    return controlled


def make_group_assignments(student_base):
    group_target = (
        student_base.groupby("id_student")["is_attrition"]
        .max()
        .sort_index()
    )
    student_ids = group_target.index.to_numpy()
    target = group_target.to_numpy()

    train_validation, test = train_test_split(
        student_ids,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    train_validation_target = group_target.loc[
        train_validation
    ].to_numpy()
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_validation_target,
    )

    assignment = {}
    for split_name, ids in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for student_id in ids:
            assignment[int(student_id)] = split_name
    return assignment


def attach_split(frame, assignments):
    split = frame["id_student"].map(assignments)
    if split.isna().any():
        raise ValueError("Some students did not receive a split.")
    output = frame.copy()
    output["split"] = split
    return output


def feature_frame(frame):
    drop_columns = [
        column
        for column in NON_FEATURE_COLUMNS + ["split"]
        if column in frame.columns
    ]
    return frame.drop(columns=drop_columns)


def split_feature_types(features):
    categorical = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric = [
        column
        for column in features.columns
        if column not in categorical
    ]
    return categorical, numeric


def make_preprocessor(features, scale_numeric):
    categorical, numeric = split_feature_types(features)
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", QuantileClipper()),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
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
                categorical,
            ),
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric,
            ),
        ]
    )


def make_logistic_pipeline(features, c_value):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(features, scale_numeric=True),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_xgboost_pipeline(features, target, params):
    positive = int((target == 1).sum())
    negative = int((target == 0).sum())
    scale_pos_weight = negative / positive if positive else 1.0

    model = XGBClassifier(
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        objective="binary:logistic",
        eval_metric="logloss",
        verbosity=0,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        **params,
    )
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(features, scale_numeric=False),
            ),
            ("classifier", model),
        ]
    )


def tune_threshold(y_true, scores):
    rows = []
    for threshold in np.round(np.arange(0.10, 0.81, 0.02), 2):
        prediction = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
                "precision": precision_score(
                    y_true,
                    prediction,
                    zero_division=0,
                ),
            }
        )
    result = pd.DataFrame(rows)
    best = result.sort_values(
        ["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"])


def select_prespecified_settings(
    selection_frame,
    output_name="setting_selection_day35.csv",
):
    train = selection_frame.loc[
        selection_frame["split"] == "train"
    ]
    validation = selection_frame.loc[
        selection_frame["split"] == "validation"
    ]

    x_train = feature_frame(train)
    y_train = train["future_withdrawal"].astype(int)
    x_validation = feature_frame(validation)
    y_validation = validation["future_withdrawal"].astype(int)

    candidate_rows = []
    for params in XGBOOST_CANDIDATES:
        model = make_xgboost_pipeline(
            x_train,
            y_train,
            params,
        )
        model.fit(x_train, y_train)
        score = model.predict_proba(x_validation)[:, 1]
        candidate_rows.append(
            {
                "model": "XGBoost",
                "settings": str(params),
                "average_precision": average_precision_score(
                    y_validation,
                    score,
                ),
                "roc_auc": roc_auc_score(y_validation, score),
                "params": params,
            }
        )

    for c_value in LOGISTIC_CANDIDATES:
        model = make_logistic_pipeline(x_train, c_value)
        model.fit(x_train, y_train)
        score = model.predict_proba(x_validation)[:, 1]
        candidate_rows.append(
            {
                "model": "Logistic Regression",
                "settings": f"C={c_value}",
                "average_precision": average_precision_score(
                    y_validation,
                    score,
                ),
                "roc_auc": roc_auc_score(y_validation, score),
                "c_value": c_value,
            }
        )

    candidates = pd.DataFrame(candidate_rows)
    candidates.drop(
        columns=["params", "c_value"],
        errors="ignore",
    ).to_csv(
        RESULTS / output_name,
        index=False,
    )

    xgb_rows = [
        row
        for row in candidate_rows
        if row["model"] == "XGBoost"
    ]
    logistic_rows = [
        row
        for row in candidate_rows
        if row["model"] == "Logistic Regression"
    ]
    best_xgb = max(
        xgb_rows,
        key=lambda row: row["average_precision"],
    )
    best_logistic = max(
        logistic_rows,
        key=lambda row: row["average_precision"],
    )
    return best_xgb["params"], best_logistic["c_value"]


def make_presentation_holdout(day75_frame):
    frame = day75_frame.copy()
    test_mask = frame["code_presentation"].eq("2014J")
    test_students = set(frame.loc[test_mask, "id_student"])

    validation_mask = (
        frame["code_presentation"].eq("2014B")
        & ~frame["id_student"].isin(test_students)
    )
    validation_students = set(
        frame.loc[validation_mask, "id_student"]
    )

    train_mask = (
        frame["code_presentation"].str.startswith("2013")
        & ~frame["id_student"].isin(test_students)
        & ~frame["id_student"].isin(validation_students)
    )

    frame["split"] = ""
    frame.loc[train_mask, "split"] = "train"
    frame.loc[validation_mask, "split"] = "validation"
    frame.loc[test_mask, "split"] = "test"
    frame = frame.loc[
        frame["split"].isin(["train", "validation", "test"])
    ].copy()
    return frame.reset_index(drop=True)


def make_uci_day_one_frame():
    path = (
        CAPSTONE
        / "work"
        / "milestone3"
        / "data"
        / "raw"
        / "uci_student_dropout.csv"
    )
    frame = pd.read_csv(path)
    semester_columns = [
        column
        for column in frame.columns
        if column.startswith("Curricular units")
    ]
    frame = frame.drop(columns=semester_columns)
    frame["future_withdrawal"] = (
        frame["Target"] == "Dropout"
    ).astype(int)
    frame = frame.drop(columns=["Target"])

    categorical_columns = [
        column
        for column in frame.columns
        if column not in UCI_CONTINUOUS_COLUMNS
        and column != "future_withdrawal"
    ]
    for column in categorical_columns:
        frame[column] = frame[column].astype("string")

    frame["id_student"] = np.arange(len(frame))
    frame["code_module"] = "UCI"
    frame["code_presentation"] = "day_one"

    train_validation, test = train_test_split(
        frame.index,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=frame["future_withdrawal"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=frame.loc[
            train_validation,
            "future_withdrawal",
        ],
    )
    frame["split"] = ""
    frame.loc[train, "split"] = "train"
    frame.loc[validation, "split"] = "validation"
    frame.loc[test, "split"] = "test"
    return frame


def calibration_summary(y_true, scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibration_model = LogisticRegression(
        penalty=None,
        max_iter=2000,
    )
    calibration_model.fit(log_odds, y_true)
    return {
        "brier_score": brier_score_loss(y_true, scores),
        "calibration_intercept": float(
            calibration_model.intercept_[0]
        ),
        "calibration_slope": float(
            calibration_model.coef_[0][0]
        ),
    }


def fit_probability_calibrator(y_true, scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(
        penalty=None,
        max_iter=2000,
    )
    calibrator.fit(log_odds, y_true)
    return calibrator


def apply_probability_calibrator(calibrator, scores):
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(log_odds)[:, 1]


def calculate_metrics(
    y_true,
    scores,
    threshold,
    training_prevalence,
):
    prediction = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, prediction)
    tn, fp, fn, tp = matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0

    null_scores = np.repeat(training_prevalence, len(y_true))
    null_brier = brier_score_loss(y_true, null_scores)
    model_brier = brier_score_loss(y_true, scores)

    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            prediction,
        ),
        "precision": precision_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(
            y_true,
            scores,
        ),
        "null_brier_score": null_brier,
        "brier_skill_score": (
            1 - model_brier / null_brier
            if null_brier
            else np.nan
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    metrics.update(calibration_summary(y_true, scores))
    return metrics


def cluster_bootstrap_intervals(
    prediction_frame,
    threshold,
    seed,
):
    rng = np.random.default_rng(seed)
    group_codes, student_ids = pd.factorize(
        prediction_frame["id_student"],
        sort=True,
    )
    group_count = len(student_ids)
    y_true = prediction_frame["y_true"].to_numpy()
    scores = prediction_frame["score"].to_numpy()
    prediction = (scores >= threshold).astype(int)
    values = {
        "f1": [],
        "roc_auc": [],
        "average_precision": [],
        "recall": [],
    }

    for _ in range(BOOTSTRAP_SAMPLES):
        group_weights = rng.multinomial(
            group_count,
            np.repeat(1 / group_count, group_count),
        )
        row_weights = group_weights[group_codes]
        positive_weight = row_weights[y_true == 1].sum()
        negative_weight = row_weights[y_true == 0].sum()
        if positive_weight == 0 or negative_weight == 0:
            continue
        values["f1"].append(
            f1_score(
                y_true,
                prediction,
                sample_weight=row_weights,
                zero_division=0,
            )
        )
        values["roc_auc"].append(
            roc_auc_score(
                y_true,
                scores,
                sample_weight=row_weights,
            )
        )
        values["average_precision"].append(
            average_precision_score(
                y_true,
                scores,
                sample_weight=row_weights,
            )
        )
        values["recall"].append(
            recall_score(
                y_true,
                prediction,
                sample_weight=row_weights,
                zero_division=0,
            )
        )

    intervals = {}
    for metric, metric_values in values.items():
        lower, upper = np.percentile(
            metric_values,
            [2.5, 97.5],
        )
        intervals[metric] = (float(lower), float(upper))
    return intervals


def train_and_evaluate(
    frame,
    cutoff,
    cohort_name,
    model_name,
    xgb_params,
    logistic_c,
):
    train = frame.loc[frame["split"] == "train"]
    validation = frame.loc[frame["split"] == "validation"]
    test = frame.loc[frame["split"] == "test"]

    x_train = feature_frame(train)
    y_train = train["future_withdrawal"].astype(int)
    x_validation = feature_frame(validation)
    y_validation = validation["future_withdrawal"].astype(int)
    x_test = feature_frame(test)
    y_test = test["future_withdrawal"].astype(int)

    if model_name == "XGBoost":
        pipeline = make_xgboost_pipeline(
            x_train,
            y_train,
            xgb_params,
        )
    else:
        pipeline = make_logistic_pipeline(
            x_train,
            logistic_c,
        )

    pipeline.fit(x_train, y_train)
    raw_validation_scores = pipeline.predict_proba(
        x_validation
    )[:, 1]
    calibrator = fit_probability_calibrator(
        y_validation,
        raw_validation_scores,
    )
    validation_scores = apply_probability_calibrator(
        calibrator,
        raw_validation_scores,
    )
    threshold = tune_threshold(
        y_validation,
        validation_scores,
    )
    capacity_threshold = float(
        np.quantile(validation_scores, 0.85)
    )
    raw_test_scores = pipeline.predict_proba(x_test)[:, 1]
    test_scores = apply_probability_calibrator(
        calibrator,
        raw_test_scores,
    )
    metrics = calculate_metrics(
        y_test,
        test_scores,
        threshold,
        training_prevalence=y_train.mean(),
    )
    capacity_metrics = calculate_metrics(
        y_test,
        test_scores,
        capacity_threshold,
        training_prevalence=y_train.mean(),
    )

    predictions = test[
        KEY_COLUMNS + ["future_withdrawal"]
    ].copy()
    predictions = predictions.rename(
        columns={"future_withdrawal": "y_true"}
    )
    predictions["raw_score"] = raw_test_scores
    predictions["score"] = test_scores
    predictions["prediction"] = (
        test_scores >= threshold
    ).astype(int)
    predictions["capacity_prediction"] = (
        test_scores >= capacity_threshold
    ).astype(int)
    predictions["cutoff_day"] = cutoff
    predictions["cohort"] = cohort_name
    predictions["model"] = model_name

    intervals = cluster_bootstrap_intervals(
        predictions,
        threshold,
        seed=RANDOM_STATE + cutoff,
    )
    row = {
        "cohort": cohort_name,
        "cutoff_day": cutoff,
        "model": model_name,
        "rows": len(frame),
        "students": frame["id_student"].nunique(),
        "positive_rows": int(frame["future_withdrawal"].sum()),
        "prevalence": frame["future_withdrawal"].mean(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "test_students": test["id_student"].nunique(),
        "threshold": threshold,
        "capacity_threshold": capacity_threshold,
        **metrics,
    }
    for metric in (
        "precision",
        "recall",
        "specificity",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ):
        row[f"capacity_{metric}"] = capacity_metrics[metric]
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper
    return pipeline, row, predictions


def clean_feature_name(name):
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            return name.replace(prefix, "", 1)
    return name


def feature_family(name):
    name = clean_feature_name(name)
    if name.startswith("vle_"):
        return "Engagement"
    if name.startswith("assessment_") or name.startswith(
        "early_assessment"
    ):
        return "Academic progress"
    if name.startswith(
        (
            "code_module",
            "code_presentation",
            "studied_credits",
            "module_presentation_length",
            "date_registration",
            "registration_lead_days",
        )
    ):
        return "Program setup"
    if name.startswith(
        ("gender", "region", "imd_band", "age_band", "disability")
    ):
        return "Demographics"
    if name.startswith(
        ("highest_education", "num_of_prev_attempts")
    ):
        return "Prior preparation"
    return "Other"


def calculate_shap(pipeline, test_frame, cutoff, cohort_name):
    x_test = feature_frame(test_frame)
    if len(x_test) > 500:
        x_sample = x_test.sample(
            n=500,
            random_state=RANDOM_STATE,
        )
    else:
        x_sample = x_test

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocessor.transform(x_sample)
    names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(classifier)
    values = np.asarray(explainer.shap_values(transformed))
    if values.ndim == 3:
        mean_absolute = np.abs(values).mean(axis=(0, 2))
    else:
        mean_absolute = np.abs(values).mean(axis=0)

    output = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": mean_absolute,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    output["feature_family"] = output["feature"].map(
        feature_family
    )
    output["cutoff_day"] = cutoff
    output["cohort"] = cohort_name
    return output


def paired_bootstrap_controlled(predictions):
    xgb = predictions.loc[
        (predictions["cohort"] == "controlled_day75")
        & (predictions["model"] == "XGBoost")
    ].copy()
    keys = KEY_COLUMNS
    windows = {}
    for cutoff in TIME_WINDOWS:
        frame = xgb.loc[xgb["cutoff_day"] == cutoff]
        windows[cutoff] = frame.set_index(keys).sort_index()

    rows = []
    rng = np.random.default_rng(RANDOM_STATE + 800)
    for earlier, later in ((35, 60), (60, 75), (35, 75)):
        first = windows[earlier]
        second = windows[later]
        common = first.index.intersection(second.index)
        first = first.loc[common]
        second = second.loc[common]
        if not np.array_equal(
            first["y_true"].to_numpy(),
            second["y_true"].to_numpy(),
        ):
            raise ValueError("Controlled targets are not aligned.")

        first_reset = first.reset_index()
        second_reset = second.reset_index()
        if not first_reset[KEY_COLUMNS].equals(
            second_reset[KEY_COLUMNS]
        ):
            raise ValueError("Controlled test rows are not aligned.")
        group_codes, student_ids = pd.factorize(
            first_reset["id_student"],
            sort=True,
        )
        group_count = len(student_ids)
        y_true = first_reset["y_true"].to_numpy()
        first_score = first_reset["score"].to_numpy()
        second_score = second_reset["score"].to_numpy()
        differences = {"roc_auc": [], "average_precision": []}

        for _ in range(BOOTSTRAP_SAMPLES):
            group_weights = rng.multinomial(
                group_count,
                np.repeat(1 / group_count, group_count),
            )
            row_weights = group_weights[group_codes]
            positive_weight = row_weights[y_true == 1].sum()
            negative_weight = row_weights[y_true == 0].sum()
            if positive_weight == 0 or negative_weight == 0:
                continue
            differences["roc_auc"].append(
                roc_auc_score(
                    y_true,
                    second_score,
                    sample_weight=row_weights,
                )
                - roc_auc_score(
                    y_true,
                    first_score,
                    sample_weight=row_weights,
                )
            )
            differences["average_precision"].append(
                average_precision_score(
                    y_true,
                    second_score,
                    sample_weight=row_weights,
                )
                - average_precision_score(
                    y_true,
                    first_score,
                    sample_weight=row_weights,
                )
            )

        for metric, values in differences.items():
            lower, upper = np.percentile(values, [2.5, 97.5])
            rows.append(
                {
                    "comparison": f"day_{later}_minus_day_{earlier}",
                    "metric": metric,
                    "mean_difference": np.mean(values),
                    "ci_95_lower": lower,
                    "ci_95_upper": upper,
                    "ci_excludes_zero": lower > 0 or upper < 0,
                }
            )
    return pd.DataFrame(rows)


def plot_performance(metrics):
    dynamic = metrics.loc[
        (metrics["cohort"] == "dynamic_landmark")
        & (metrics["model"] == "XGBoost")
    ].sort_values("cutoff_day")
    controlled = metrics.loc[
        (metrics["cohort"] == "controlled_day75")
        & (metrics["model"] == "XGBoost")
    ].sort_values("cutoff_day")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for frame, label, color in (
        (dynamic, "Operational landmark cohort", "#247B7B"),
        (controlled, "Fixed day-75 cohort", "#C66A32"),
    ):
        axes[0].plot(
            frame["cutoff_day"],
            frame["roc_auc"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
        axes[1].plot(
            frame["cutoff_day"],
            frame["average_precision"],
            marker="o",
            linewidth=2,
            label=label,
            color=color,
        )
    axes[0].set_title("ROC-AUC by Observation Window")
    axes[1].set_title("Average Precision by Observation Window")
    for axis in axes:
        axis.set_xlabel("Course day")
        axis.set_xticks(TIME_WINDOWS)
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "journal_temporal_performance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_calibration(predictions):
    selected = predictions.loc[
        (predictions["cohort"] == "dynamic_landmark")
        & (predictions["model"] == "XGBoost")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4))

    for axis, cutoff in zip(axes, TIME_WINDOWS):
        frame = selected.loc[
            selected["cutoff_day"] == cutoff
        ].copy()
        frame["bin"] = pd.qcut(
            frame["score"],
            q=10,
            duplicates="drop",
        )
        calibration = frame.groupby(
            "bin",
            observed=True,
        ).agg(
            mean_prediction=("score", "mean"),
            observed_rate=("y_true", "mean"),
        )
        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
        )
        axis.plot(
            calibration["mean_prediction"],
            calibration["observed_rate"],
            marker="o",
            color="#247B7B",
        )
        axis.set_title(f"Day {cutoff}")
        axis.set_xlabel("Mean predicted risk")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
    axes[0].set_ylabel("Observed withdrawal rate")
    fig.suptitle("Calibration on Group-Held-Out Test Students")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "journal_calibration.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_shap_families(shap_results):
    grouped = (
        shap_results.groupby(
            ["cohort", "cutoff_day", "feature_family"],
            as_index=False,
        )["mean_abs_shap"]
        .sum()
    )
    grouped["share"] = grouped.groupby(
        ["cohort", "cutoff_day"]
    )["mean_abs_shap"].transform(
        lambda values: values / values.sum()
    )
    grouped.to_csv(
        RESULTS / "journal_shap_families.csv",
        index=False,
    )

    controlled = grouped.loc[
        grouped["cohort"] == "controlled_day75"
    ]
    pivot = controlled.pivot_table(
        index="feature_family",
        columns="cutoff_day",
        values="share",
        fill_value=0,
    )
    pivot = pivot.reindex(columns=TIME_WINDOWS).fillna(0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    pivot.plot(kind="barh", ax=ax)
    ax.set_title("Feature-Family Importance in the Fixed Day-75 Cohort")
    ax.set_xlabel("Share of mean absolute SHAP")
    ax.set_ylabel("")
    ax.legend(
        [f"Day {cutoff}" for cutoff in TIME_WINDOWS],
        title="Observation window",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "journal_shap_family_shift.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def check_split_overlap(
    frames,
    output_name="split_overlap_check.csv",
):
    rows = []
    for cohort_name, cutoff, frame in frames:
        students = {
            split_name: set(
                frame.loc[
                    frame["split"] == split_name,
                    "id_student",
                ]
            )
            for split_name in ("train", "validation", "test")
        }
        rows.append(
            {
                "cohort": cohort_name,
                "cutoff_day": cutoff,
                "train_validation_overlap": len(
                    students["train"] & students["validation"]
                ),
                "train_test_overlap": len(
                    students["train"] & students["test"]
                ),
                "validation_test_overlap": len(
                    students["validation"] & students["test"]
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(
        RESULTS / output_name,
        index=False,
    )
    if result.filter(like="overlap").to_numpy().max() != 0:
        raise ValueError("Student overlap remains across splits.")


def main():
    ensure_dirs()
    print("Step 1: Build cutoff-specific OULAD features")
    tables = temporal.read_oulad_tables()
    student_base = build_student_base(tables)
    window_frames = build_window_frames(student_base, tables)

    print("Step 2: Define future-withdrawal landmark cohorts")
    dynamic = {
        cutoff: make_landmark_cohort(
            window_frames[cutoff],
            cutoff,
        )
        for cutoff in TIME_WINDOWS
    }
    controlled = make_controlled_cohorts(
        window_frames,
        dynamic[75],
    )

    assignments = make_group_assignments(student_base)
    for cutoff in TIME_WINDOWS:
        dynamic[cutoff] = attach_split(
            dynamic[cutoff],
            assignments,
        )
        controlled[cutoff] = attach_split(
            controlled[cutoff],
            assignments,
        )

    frames_for_check = []
    for cohort_name, cohort_frames in (
        ("dynamic_landmark", dynamic),
        ("controlled_day75", controlled),
    ):
        for cutoff, frame in cohort_frames.items():
            frames_for_check.append(
                (cohort_name, cutoff, frame)
            )
    check_split_overlap(frames_for_check)

    print("Step 3: Select settings using only day-35 validation data")
    xgb_params, logistic_c = select_prespecified_settings(
        dynamic[35]
    )
    print("Selected XGBoost settings:", xgb_params)
    print("Selected logistic C:", logistic_c)

    print("Step 4: Train group-separated landmark models")
    metric_rows = []
    prediction_frames = []
    shap_frames = []

    for cohort_name, cohort_frames in (
        ("dynamic_landmark", dynamic),
        ("controlled_day75", controlled),
    ):
        for cutoff in TIME_WINDOWS:
            frame = cohort_frames[cutoff]
            for model_name in (
                "Logistic Regression",
                "XGBoost",
            ):
                print(cohort_name, cutoff, model_name)
                model, metrics, predictions = train_and_evaluate(
                    frame,
                    cutoff,
                    cohort_name,
                    model_name,
                    xgb_params,
                    logistic_c,
                )
                metric_rows.append(metrics)
                prediction_frames.append(predictions)

                if model_name == "XGBoost":
                    test_frame = frame.loc[
                        frame["split"] == "test"
                    ]
                    shap_frames.append(
                        calculate_shap(
                            model,
                            test_frame,
                            cutoff,
                            cohort_name,
                        )
                    )

    print("Step 5: Test a later-presentation holdout")
    presentation_holdout = make_presentation_holdout(
        dynamic[75]
    )
    check_split_overlap(
        [
            (
                "presentation_holdout_2014J",
                75,
                presentation_holdout,
            )
        ],
        output_name="split_overlap_presentation_holdout.csv",
    )
    external_xgb_params, external_logistic_c = (
        select_prespecified_settings(
            presentation_holdout,
            output_name="setting_selection_presentation_holdout.csv",
        )
    )
    for model_name in ("Logistic Regression", "XGBoost"):
        model, metrics, predictions = train_and_evaluate(
            presentation_holdout,
            75,
            "presentation_holdout_2014J",
            model_name,
            external_xgb_params,
            external_logistic_c,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)
        if model_name == "XGBoost":
            test_frame = presentation_holdout.loc[
                presentation_holdout["split"] == "test"
            ]
            shap_frames.append(
                calculate_shap(
                    model,
                    test_frame,
                    75,
                    "presentation_holdout_2014J",
                )
            )

    print("Step 6: Run the UCI day-one benchmark")
    uci_day_one = make_uci_day_one_frame()
    uci_xgb_params, uci_logistic_c = select_prespecified_settings(
        uci_day_one,
        output_name="setting_selection_uci_day_one.csv",
    )
    for model_name in ("Logistic Regression", "XGBoost"):
        model, metrics, predictions = train_and_evaluate(
            uci_day_one,
            0,
            "uci_day_one",
            model_name,
            uci_xgb_params,
            uci_logistic_c,
        )
        metric_rows.append(metrics)
        prediction_frames.append(predictions)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    shap_results = pd.concat(
        shap_frames,
        ignore_index=True,
    )

    metrics.to_csv(
        RESULTS / "journal_model_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        RESULTS / "journal_test_predictions.csv",
        index=False,
    )
    shap_results.to_csv(
        RESULTS / "journal_shap_features.csv",
        index=False,
    )

    print("Step 7: Compare fixed-cohort windows")
    paired = paired_bootstrap_controlled(predictions)
    paired.to_csv(
        RESULTS / "journal_paired_differences.csv",
        index=False,
    )

    print("Step 8: Save journal figures")
    plot_performance(metrics)
    plot_calibration(predictions)
    plot_shap_families(shap_results)

    cohort_rows = []
    for cutoff, frame in dynamic.items():
        cohort_rows.append(
            {
                "cohort": "dynamic_landmark",
                "cutoff_day": cutoff,
                "rows": len(frame),
                "students": frame["id_student"].nunique(),
                "future_withdrawals": int(
                    frame["future_withdrawal"].sum()
                ),
                "prevalence": frame["future_withdrawal"].mean(),
            }
        )
    for cutoff, frame in controlled.items():
        cohort_rows.append(
            {
                "cohort": "controlled_day75",
                "cutoff_day": cutoff,
                "rows": len(frame),
                "students": frame["id_student"].nunique(),
                "future_withdrawals": int(
                    frame["future_withdrawal"].sum()
                ),
                "prevalence": frame["future_withdrawal"].mean(),
            }
        )
    cohort_rows.append(
        {
            "cohort": "presentation_holdout_2014J",
            "cutoff_day": 75,
            "rows": len(presentation_holdout),
            "students": presentation_holdout[
                "id_student"
            ].nunique(),
            "future_withdrawals": int(
                presentation_holdout[
                    "future_withdrawal"
                ].sum()
            ),
            "prevalence": presentation_holdout[
                "future_withdrawal"
            ].mean(),
        }
    )
    cohort_rows.append(
        {
            "cohort": "uci_day_one",
            "cutoff_day": 0,
            "rows": len(uci_day_one),
            "students": len(uci_day_one),
            "future_withdrawals": int(
                uci_day_one["future_withdrawal"].sum()
            ),
            "prevalence": uci_day_one[
                "future_withdrawal"
            ].mean(),
        }
    )
    pd.DataFrame(cohort_rows).to_csv(
        RESULTS / "journal_cohort_summary.csv",
        index=False,
    )

    print("\nMain operational results")
    print(
        metrics.loc[
            metrics["cohort"] == "dynamic_landmark",
            [
                "cutoff_day",
                "model",
                "rows",
                "prevalence",
                "threshold",
                "f1",
                "roc_auc",
                "average_precision",
                "brier_score",
            ],
        ].round(4)
    )


if __name__ == "__main__":
    main()
