from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import feature_preparation as temporal


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
RAW = PACKAGE / "data" / "raw"
RESULTS = PACKAGE / "outputs" / "results"
FIGURES = PACKAGE / "outputs" / "figures"
MODELS = PACKAGE / "models"

MODEL_PATH = MODELS / "unified_dynamic_xgboost.joblib"
CONTRACT_PATH = MODELS / "unified_dynamic_feature_contract.json"

RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 1000
PERMUTATION_SAMPLES = 1000
SCHEMA_VERSION = "3.0"
SNAPSHOT_DAYS = [0, 35, 60, 75]
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]

# Every model input is a numeric semantic concept on a documented 0-1 scale.
# Raw course identifiers, regions, assignment numbers, and LMS activity types
# are intentionally excluded because they do not have stable meanings across
# institutions.
CORE_FEATURES = [
    "age_scaled",
    "prior_education_level",
    "study_load",
    "course_progress_ratio",
]

BACKGROUND_FEATURES = [
    "prior_academic_score",
    "previous_attempts",
    "male",
    "declared_support_need",
    "financial_stability",
]

ASSESSMENT_FEATURES = [
    "assessment_completion_rate",
    "assessment_average_score",
    "assessment_score_available",
    "late_submission_rate",
]

ACTIVITY_FEATURES = [
    "active_day_rate",
    "days_since_last_activity_scaled",
    "recent_activity_rate",
]

OPTIONAL_FEATURES = (
    BACKGROUND_FEATURES + ASSESSMENT_FEATURES + ACTIVITY_FEATURES
)
DYNAMIC_FEATURES = ASSESSMENT_FEATURES + ACTIVITY_FEATURES
MODEL_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES

UCI_EDUCATION = {
    38: 0.15,
    19: 0.25,
    15: 0.30,
    14: 0.35,
    12: 0.40,
    10: 0.40,
    9: 0.45,
    1: 0.55,
    39: 0.65,
    42: 0.70,
    6: 0.70,
    2: 0.75,
    40: 0.80,
    3: 0.82,
    43: 0.92,
    4: 0.92,
    5: 1.00,
}

OULAD_EDUCATION = {
    "No Formal quals": 0.10,
    "Lower Than A Level": 0.30,
    "A Level or Equivalent": 0.55,
    "HE Qualification": 0.80,
    "Post Graduate Qualification": 1.00,
}

SCHEMA_ROWS = [
    {
        "feature": "age_scaled",
        "required": True,
        "group": "background",
        "meaning": "Age position on a fixed 18-60 scale",
        "uci_mapping": "(Age at enrollment - 18) / 42",
        "oulad_mapping": "Age-band midpoint on the same scale",
        "new_school_requirement": "Age or a compatible age band",
    },
    {
        "feature": "prior_education_level",
        "required": True,
        "group": "background",
        "meaning": "Highest prior education on a documented 0-1 ladder",
        "uci_mapping": "Previous qualification",
        "oulad_mapping": "highest_education",
        "new_school_requirement": "Local education ladder mapped to 0-1",
    },
    {
        "feature": "study_load",
        "required": True,
        "group": "background",
        "meaning": "Registered load relative to a heavy full load",
        "uci_mapping": "First-semester enrolled units / 12",
        "oulad_mapping": "studied_credits / 120",
        "new_school_requirement": "Registered load / documented heavy load",
    },
    {
        "feature": "course_progress_ratio",
        "required": True,
        "group": "timing",
        "meaning": "Share of planned course duration elapsed",
        "uci_mapping": "0 at enrolment",
        "oulad_mapping": "cutoff day / presentation length",
        "new_school_requirement": "Elapsed days / planned course days",
    },
    {
        "feature": "prior_academic_score",
        "required": False,
        "group": "background",
        "meaning": "Normalized pre-entry grade or GPA",
        "uci_mapping": "Previous qualification grade / 200",
        "oulad_mapping": "Unavailable",
        "new_school_requirement": "Prior score / local maximum or blank",
    },
    {
        "feature": "previous_attempts",
        "required": False,
        "group": "background",
        "meaning": "Prior attempts capped and scaled to 0-1",
        "uci_mapping": "Unavailable",
        "oulad_mapping": "num_of_prev_attempts / 3",
        "new_school_requirement": "Prior attempts / documented cap or blank",
    },
    {
        "feature": "male",
        "required": False,
        "group": "sensitive_optional",
        "meaning": "Optional documented binary gender indicator",
        "uci_mapping": "Gender",
        "oulad_mapping": "gender",
        "new_school_requirement": "0/1 if approved; otherwise blank",
    },
    {
        "feature": "declared_support_need",
        "required": False,
        "group": "sensitive_optional",
        "meaning": "Optional declared disability or support need",
        "uci_mapping": "Educational special needs",
        "oulad_mapping": "disability",
        "new_school_requirement": "0/1 if approved; otherwise blank",
    },
    {
        "feature": "financial_stability",
        "required": False,
        "group": "sensitive_optional",
        "meaning": "Optional documented financial-stability proxy",
        "uci_mapping": "Debt and tuition status",
        "oulad_mapping": "Area-deprivation-band midpoint",
        "new_school_requirement": "Comparable 0-1 measure or blank",
    },
    {
        "feature": "assessment_completion_rate",
        "required": False,
        "group": "assessment",
        "meaning": "Share of due assessment weight submitted by snapshot",
        "uci_mapping": "Unavailable at enrolment",
        "oulad_mapping": "Submitted due weight / total due weight",
        "new_school_requirement": "Same due-by-snapshot calculation or blank",
    },
    {
        "feature": "assessment_average_score",
        "required": False,
        "group": "assessment",
        "meaning": "Weighted normalized score among submitted assessments",
        "uci_mapping": "Unavailable at enrolment",
        "oulad_mapping": "Weighted submitted score / submitted weight",
        "new_school_requirement": "Earned / possible score through snapshot",
    },
    {
        "feature": "assessment_score_available",
        "required": False,
        "group": "assessment",
        "meaning": "Whether at least one score is available at snapshot",
        "uci_mapping": "0 at enrolment",
        "oulad_mapping": "Any submitted scored assessment by snapshot",
        "new_school_requirement": "0/1 or blank if feed is unavailable",
    },
    {
        "feature": "late_submission_rate",
        "required": False,
        "group": "assessment",
        "meaning": "Share of submitted assessments submitted after due date",
        "uci_mapping": "Unavailable",
        "oulad_mapping": "Late submissions / submissions by snapshot",
        "new_school_requirement": "Same due-date calculation or blank",
    },
    {
        "feature": "active_day_rate",
        "required": False,
        "group": "activity",
        "meaning": "Share of observable course days with LMS activity",
        "uci_mapping": "Unavailable",
        "oulad_mapping": "Active days / days observable by snapshot",
        "new_school_requirement": "Platform-neutral active-day rate or blank",
    },
    {
        "feature": "days_since_last_activity_scaled",
        "required": False,
        "group": "activity",
        "meaning": "Days since last LMS activity capped at 30 and divided by 30",
        "uci_mapping": "Unavailable",
        "oulad_mapping": "Snapshot day minus last activity day",
        "new_school_requirement": "Platform-neutral recency or blank",
    },
    {
        "feature": "recent_activity_rate",
        "required": False,
        "group": "activity",
        "meaning": "Share of observable days active in the last 14 days",
        "uci_mapping": "Unavailable",
        "oulad_mapping": "Active days in trailing 14-day window / days",
        "new_school_requirement": "Platform-neutral trailing activity or blank",
    },
]


def imd_midpoint(values):
    extracted = values.astype("string").str.extract(r"(\d+)-(\d+)%")
    low = pd.to_numeric(extracted[0], errors="coerce")
    high = pd.to_numeric(extracted[1], errors="coerce")
    return (low + high) / 200


def make_uci_snapshots():
    raw = pd.read_csv(RAW / "uci_student_dropout.csv")
    frame = pd.DataFrame(index=raw.index)
    frame["institution"] = "UCI_dataset"
    frame["record_id"] = "UCI_" + raw.index.astype(str) + "_D0"
    frame["student_id"] = "UCI_" + raw.index.astype(str)
    frame["course_id"] = raw["Course"].astype(str)
    frame["snapshot_day"] = 0
    frame["target"] = raw["Target"].eq("Dropout").astype(int)

    frame["age_scaled"] = (
        (raw["Age at enrollment"] - 18) / 42
    ).clip(0, 1)
    frame["prior_education_level"] = raw[
        "Previous qualification"
    ].map(UCI_EDUCATION)
    frame["prior_academic_score"] = (
        raw["Previous qualification (grade)"] / 200
    ).clip(0, 1)
    frame["previous_attempts"] = np.nan
    frame["study_load"] = (
        raw["Curricular units 1st sem (enrolled)"] / 12
    ).clip(0, 1)
    frame["male"] = pd.to_numeric(raw["Gender"], errors="coerce")
    frame["declared_support_need"] = pd.to_numeric(
        raw["Educational special needs"],
        errors="coerce",
    )
    frame["financial_stability"] = (
        (1 - raw["Debtor"].astype(float))
        + raw["Tuition fees up to date"].astype(float)
    ) / 2
    frame["course_progress_ratio"] = 0.0
    for feature in DYNAMIC_FEATURES:
        frame[feature] = np.nan
    frame["assessment_score_available"] = 0.0
    return frame.reset_index(drop=True)


def build_oulad_student_base(tables):
    base = tables["student_info"].merge(
        tables["student_registration"],
        on=KEY_COLUMNS,
        how="left",
    )
    base = base.merge(
        tables["courses"],
        on=["code_module", "code_presentation"],
        how="left",
    )
    return base


def make_assessment_snapshot(tables, cutoff):
    assessments = tables["assessments"].loc[
        tables["assessments"]["date"].notna()
        & tables["assessments"]["date"].le(cutoff)
    ].copy()
    schedule = (
        assessments.groupby(
            ["code_module", "code_presentation"],
            as_index=False,
        )
        .agg(
            due_count=("id_assessment", "nunique"),
            due_weight=("weight", "sum"),
        )
    )

    submitted = tables["assessment_base"].loc[
        tables["assessment_base"]["date"].notna()
        & tables["assessment_base"]["date"].le(cutoff)
        & tables["assessment_base"]["date_submitted"].notna()
        & tables["assessment_base"]["date_submitted"].le(cutoff)
    ].copy()
    submitted["submitted_weight"] = submitted["weight"].fillna(0.0)
    submitted["weighted_normalized_score"] = (
        submitted["score"].fillna(0.0)
        / 100.0
        * submitted["submitted_weight"]
    )
    submitted["late"] = (
        submitted["date_submitted"] > submitted["date"]
    ).astype(float)
    student = (
        submitted.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            submitted_count=("id_assessment", "nunique"),
            submitted_weight=("submitted_weight", "sum"),
            weighted_score=("weighted_normalized_score", "sum"),
            mean_score=("score", "mean"),
            late_count=("late", "sum"),
        )
    )
    return schedule, student


def make_activity_snapshot(tables, cutoff):
    early = tables["student_vle"].loc[
        tables["student_vle"]["date"].le(cutoff)
    ]
    activity = (
        early.groupby(KEY_COLUMNS, as_index=False, observed=True)
        .agg(
            active_days=("date", "nunique"),
            last_activity_day=("date", "max"),
        )
    )
    recent = early.loc[early["date"].gt(cutoff - 14)]
    recent_activity = (
        recent.groupby(KEY_COLUMNS, as_index=False, observed=True)
        .agg(recent_active_days=("date", "nunique"))
    )
    for frame in (activity, recent_activity):
        frame["code_module"] = frame["code_module"].astype(str)
        frame["code_presentation"] = frame[
            "code_presentation"
        ].astype(str)
    return activity, recent_activity


def eligible_oulad_snapshot(base, cutoff):
    withdrawal_timing_known = ~(
        base["final_result"].eq("Withdrawn")
        & base["date_unregistration"].isna()
    )
    registered = (
        base["date_registration"].notna()
        & base["date_registration"].le(cutoff)
    )
    still_registered = (
        base["date_unregistration"].isna()
        | base["date_unregistration"].gt(cutoff)
    )
    frame = base.loc[
        withdrawal_timing_known & registered & still_registered
    ].copy()
    frame["target"] = (
        frame["final_result"].eq("Withdrawn")
        & frame["date_unregistration"].gt(cutoff)
    ).astype(int)
    return frame


def make_oulad_snapshot(base, tables, cutoff):
    raw = eligible_oulad_snapshot(base, cutoff)

    if cutoff > 0:
        schedule, assessment = make_assessment_snapshot(tables, cutoff)
        activity, recent_activity = make_activity_snapshot(tables, cutoff)
        raw = raw.merge(
            schedule,
            on=["code_module", "code_presentation"],
            how="left",
        )
        raw = raw.merge(assessment, on=KEY_COLUMNS, how="left")
        raw = raw.merge(activity, on=KEY_COLUMNS, how="left")
        raw = raw.merge(recent_activity, on=KEY_COLUMNS, how="left")

    frame = pd.DataFrame(index=raw.index)
    frame["institution"] = "OULAD_dataset"
    frame["record_id"] = (
        "OULAD_"
        + raw["code_module"].astype(str)
        + "_"
        + raw["code_presentation"].astype(str)
        + "_"
        + raw["id_student"].astype(str)
        + f"_D{cutoff}"
    )
    frame["student_id"] = "OULAD_" + raw["id_student"].astype(str)
    frame["course_id"] = (
        raw["code_module"].astype(str)
        + "_"
        + raw["code_presentation"].astype(str)
    )
    frame["snapshot_day"] = cutoff
    frame["target"] = raw["target"].astype(int)

    age_midpoint = raw["age_band"].map(
        {"0-35": 26.0, "35-55": 45.0, "55<=": 60.0}
    )
    frame["age_scaled"] = ((age_midpoint - 18) / 42).clip(0, 1)
    frame["prior_education_level"] = raw[
        "highest_education"
    ].map(OULAD_EDUCATION)
    frame["prior_academic_score"] = np.nan
    frame["previous_attempts"] = (
        raw["num_of_prev_attempts"] / 3
    ).clip(0, 1)
    frame["study_load"] = (raw["studied_credits"] / 120).clip(0, 1)
    frame["male"] = raw["gender"].map({"M": 1.0, "F": 0.0})
    frame["declared_support_need"] = raw["disability"].map(
        {"Y": 1.0, "N": 0.0}
    )
    frame["financial_stability"] = imd_midpoint(raw["imd_band"])
    frame["course_progress_ratio"] = (
        cutoff / pd.to_numeric(raw["module_presentation_length"])
    ).clip(0, 1)

    if cutoff == 0:
        for feature in DYNAMIC_FEATURES:
            frame[feature] = np.nan
        frame["assessment_score_available"] = 0.0
        return frame.reset_index(drop=True)

    due_count = raw["due_count"].fillna(0.0)
    due_weight = raw["due_weight"].fillna(0.0)
    submitted_count = raw["submitted_count"].fillna(0.0)
    submitted_weight = raw["submitted_weight"].fillna(0.0)

    count_completion = np.where(
        due_count > 0,
        submitted_count / due_count,
        np.nan,
    )
    weighted_completion = np.where(
        due_weight > 0,
        submitted_weight / due_weight,
        count_completion,
    )
    frame["assessment_completion_rate"] = np.clip(
        weighted_completion,
        0,
        1,
    )
    weighted_average = np.where(
        submitted_weight > 0,
        raw["weighted_score"].fillna(0.0) / submitted_weight,
        raw["mean_score"] / 100.0,
    )
    frame["assessment_average_score"] = np.where(
        submitted_count > 0,
        np.clip(weighted_average, 0, 1),
        np.nan,
    )
    frame["assessment_score_available"] = (
        submitted_count > 0
    ).astype(float)
    frame["late_submission_rate"] = np.where(
        submitted_count > 0,
        raw["late_count"].fillna(0.0) / submitted_count,
        np.nan,
    )

    observable_start = pd.to_numeric(
        raw["date_registration"],
        errors="coerce",
    ).fillna(0.0)
    observable_days = np.maximum(cutoff - observable_start + 1, 1)
    active_days = raw["active_days"].fillna(0.0)
    frame["active_day_rate"] = np.clip(
        active_days / observable_days,
        0,
        1,
    )
    days_since = cutoff - raw["last_activity_day"]
    frame["days_since_last_activity_scaled"] = (
        days_since.fillna(30.0).clip(0, 30) / 30
    )
    recent_window = np.minimum(observable_days, 14)
    frame["recent_activity_rate"] = np.clip(
        raw["recent_active_days"].fillna(0.0) / recent_window,
        0,
        1,
    )
    return frame.reset_index(drop=True)


def make_oulad_snapshots():
    tables = temporal.read_oulad_tables()
    base = build_oulad_student_base(tables)
    frames = [
        make_oulad_snapshot(base, tables, cutoff)
        for cutoff in SNAPSHOT_DAYS
    ]
    return pd.concat(frames, ignore_index=True)


def validate_frame(frame, require_target=True):
    required = {
        "institution",
        "record_id",
        "student_id",
        "snapshot_day",
        *CORE_FEATURES,
    }
    if require_target:
        required.add("target")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id values must be unique.")
    if frame[CORE_FEATURES].isna().any().any():
        missing_core = frame[CORE_FEATURES].columns[
            frame[CORE_FEATURES].isna().any()
        ].tolist()
        raise ValueError(f"Required features contain nulls: {missing_core}")
    if require_target and not frame["target"].isin([0, 1]).all():
        raise ValueError("target must contain only 0 and 1.")

    for feature in MODEL_FEATURES:
        if feature not in frame:
            if feature in OPTIONAL_FEATURES:
                continue
            raise ValueError(f"Required feature is missing: {feature}")
        values = pd.to_numeric(frame[feature], errors="coerce")
        observed = values.dropna()
        if not observed.between(0, 1).all():
            raise ValueError(f"{feature} must be between 0 and 1.")


def prepare_scoring_frame(frame):
    output = frame.copy()
    if "record_id" not in output:
        output["record_id"] = "ROW_" + output.index.astype(str)
    if "student_id" not in output:
        output["student_id"] = output["record_id"].astype(str)
    if "institution" not in output:
        output["institution"] = "new_school"
    if "snapshot_day" not in output:
        output["snapshot_day"] = np.nan
    for feature in OPTIONAL_FEATURES:
        if feature not in output:
            output[feature] = np.nan
    for feature in MODEL_FEATURES:
        if feature in output:
            output[feature] = pd.to_numeric(
                output[feature],
                errors="coerce",
            )
    validate_frame(output, require_target=False)
    return output


def assign_splits(frame, seed):
    target_by_student = frame.groupby("student_id")["target"].max()
    students = target_by_student.index.to_numpy()
    labels = target_by_student.to_numpy()
    train_validation, test = train_test_split(
        students,
        test_size=0.20,
        random_state=seed,
        stratify=labels,
    )
    remaining_labels = target_by_student.loc[train_validation].to_numpy()
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=remaining_labels,
    )
    assignments = {student: "train" for student in train}
    assignments.update({student: "validation" for student in validation})
    assignments.update({student: "test" for student in test})
    output = frame.copy()
    output["split"] = output["student_id"].map(assignments)
    return output


def mask_optional_groups(frame, seed=RANDOM_STATE):
    output = frame.copy()
    rng = np.random.default_rng(seed)
    masking = [
        (BACKGROUND_FEATURES, 0.10),
        (ASSESSMENT_FEATURES, 0.20),
        (ACTIVITY_FEATURES, 0.20),
    ]
    for features, probability in masking:
        selected = rng.random(len(output)) < probability
        output.loc[selected, features] = np.nan
    return output


def row_weights(frame, balance_classes):
    rows_per_student = frame.groupby(
        ["institution", "student_id"]
    )["record_id"].transform("size")
    weights = 1.0 / rows_per_student.to_numpy(dtype=float)

    institutions = frame["institution"].unique()
    target = frame["target"].to_numpy(dtype=int)
    if balance_classes:
        cell_share = 1.0 / (len(institutions) * 2)
        for institution in institutions:
            for value in (0, 1):
                selected = (
                    frame["institution"].eq(institution).to_numpy()
                    & (target == value)
                )
                if selected.any():
                    weights[selected] *= cell_share / weights[selected].sum()
    else:
        for institution in institutions:
            selected = frame["institution"].eq(institution).to_numpy()
            weights[selected] /= weights[selected].sum()
            weights[selected] /= len(institutions)

    weights *= len(weights) / weights.sum()
    return weights


def make_model(model_name):
    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                ("imputer", imputer),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.2,
                        max_iter=3000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    return Pipeline(
        [
            ("imputer", imputer),
            (
                "model",
                XGBClassifier(
                    n_estimators=320,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=4,
                    reg_lambda=2.0,
                    reg_alpha=0.1,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_model(train, model_name):
    masked = mask_optional_groups(train)
    model = make_model(model_name)
    model.fit(
        masked[MODEL_FEATURES],
        masked["target"],
        model__sample_weight=row_weights(train, balance_classes=True),
    )
    return model


def tune_thresholds(model, validation):
    rows = []
    for snapshot_day, group in validation.groupby("snapshot_day"):
        scores = model.predict_proba(group[MODEL_FEATURES])[:, 1]
        weights = row_weights(group, balance_classes=False)
        for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
            prediction = (scores >= threshold).astype(int)
            rows.append(
                {
                    "snapshot_day": int(snapshot_day),
                    "threshold": float(threshold),
                    "weighted_f1": f1_score(
                        group["target"],
                        prediction,
                        sample_weight=weights,
                        zero_division=0,
                    ),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["weighted_f1", "threshold"],
        ascending=[False, True],
    )
    best = (
        result.groupby("snapshot_day", as_index=False)
        .first()
        .set_index("snapshot_day")["threshold"]
        .to_dict()
    )
    return best, result


def safe_auc(target, scores):
    if pd.Series(target).nunique() < 2:
        return np.nan
    return roc_auc_score(target, scores)


def evaluate_group(
    model,
    frame,
    model_name,
    scenario,
    threshold,
):
    scores = model.predict_proba(frame[MODEL_FEATURES])[:, 1]
    prediction = (scores >= threshold).astype(int)
    metrics = {
        "feature_set": "portable_dynamic_16",
        "model": model_name,
        "training_scenario": scenario,
        "test_institution": frame["institution"].iloc[0],
        "snapshot_day": int(frame["snapshot_day"].iloc[0]),
        "test_rows": len(frame),
        "test_students": frame["student_id"].nunique(),
        "prevalence": frame["target"].mean(),
        "threshold": threshold,
        "roc_auc": safe_auc(frame["target"], scores),
        "average_precision": average_precision_score(
            frame["target"],
            scores,
        ),
        "precision": precision_score(
            frame["target"],
            prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            frame["target"],
            prediction,
            zero_division=0,
        ),
        "f1": f1_score(
            frame["target"],
            prediction,
            zero_division=0,
        ),
    }
    predictions = frame[
        [
            "institution",
            "record_id",
            "student_id",
            "course_id",
            "snapshot_day",
            "target",
        ]
    ].copy()
    predictions["model"] = model_name
    predictions["training_scenario"] = scenario
    predictions["score"] = scores
    predictions["prediction"] = prediction
    return metrics, predictions


def scenario_frames(uci, oulad):
    datasets = {"UCI_dataset": uci, "OULAD_dataset": oulad}
    definitions = {
        "uci_source": ["UCI_dataset"],
        "oulad_source": ["OULAD_dataset"],
        "pooled": ["UCI_dataset", "OULAD_dataset"],
    }
    scenarios = []
    for name, institutions in definitions.items():
        selected = [datasets[value] for value in institutions]
        combined = pd.concat(selected, ignore_index=True)
        scenarios.append(
            {
                "name": name,
                "train": combined.loc[combined["split"].eq("train")],
                "validation": combined.loc[
                    combined["split"].eq("validation")
                ],
            }
        )
    return scenarios


def test_groups(uci, oulad):
    tests = pd.concat(
        [
            uci.loc[uci["split"].eq("test")],
            oulad.loc[oulad["split"].eq("test")],
        ],
        ignore_index=True,
    )
    return [
        group.copy()
        for _, group in tests.groupby(
            ["institution", "snapshot_day"],
            sort=True,
        )
    ]


def run_experiments(uci, oulad):
    metrics = []
    predictions = []
    thresholds = []
    final_model = None
    final_threshold = None

    for model_name in ("Logistic Regression", "XGBoost"):
        for scenario in scenario_frames(uci, oulad):
            print(model_name, scenario["name"])
            model = fit_model(scenario["train"], model_name)
            threshold_by_day, threshold_table = tune_thresholds(
                model,
                scenario["validation"],
            )
            threshold_table["model"] = model_name
            threshold_table["training_scenario"] = scenario["name"]
            thresholds.append(threshold_table)

            for group in test_groups(uci, oulad):
                snapshot_day = int(group["snapshot_day"].iloc[0])
                threshold = threshold_by_day.get(
                    snapshot_day,
                    threshold_by_day.get(
                        0,
                        next(iter(threshold_by_day.values())),
                    ),
                )
                row, scored = evaluate_group(
                    model,
                    group,
                    model_name,
                    scenario["name"],
                    threshold,
                )
                row["train_rows"] = len(scenario["train"])
                row["train_students"] = scenario["train"][
                    "student_id"
                ].nunique()
                metrics.append(row)
                predictions.append(scored)

            if model_name == "XGBoost" and scenario["name"] == "pooled":
                final_model = model
                final_threshold = threshold_by_day

    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True),
        pd.concat(thresholds, ignore_index=True),
        final_model,
        final_threshold,
    )


def availability_summary(frame):
    rows = []
    for (institution, day), group in frame.groupby(
        ["institution", "snapshot_day"]
    ):
        for feature in MODEL_FEATURES:
            rows.append(
                {
                    "institution": institution,
                    "snapshot_day": day,
                    "feature": feature,
                    "available_share": group[feature].notna().mean(),
                }
            )
    return pd.DataFrame(rows)


def feature_importance(model):
    imputer = model.named_steps["imputer"]
    names = imputer.get_feature_names_out(MODEL_FEATURES)
    values = model.named_steps["model"].feature_importances_
    return pd.DataFrame(
        {"transformed_feature": names, "importance": values}
    ).sort_values("importance", ascending=False)


def metric_value(metric, target, scores, weights=None):
    if metric == "roc_auc":
        return roc_auc_score(target, scores, sample_weight=weights)
    return average_precision_score(
        target,
        scores,
        sample_weight=weights,
    )


def paired_cluster_test(first, second, comparison, seed):
    left = first[
        ["record_id", "student_id", "target", "score"]
    ].rename(columns={"score": "first_score"})
    right = second[["record_id", "target", "score"]].rename(
        columns={"target": "second_target", "score": "second_score"}
    )
    paired = left.merge(right, on="record_id", how="inner")
    if paired.empty:
        raise ValueError(f"No paired rows were found for {comparison}.")
    if not paired["target"].equals(paired["second_target"]):
        raise ValueError(f"Targets do not match for {comparison}.")

    group_codes, students = pd.factorize(
        paired["student_id"],
        sort=True,
    )
    target = paired["target"].to_numpy()
    first_score = paired["first_score"].to_numpy()
    second_score = paired["second_score"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = []

    for metric in ("roc_auc", "average_precision"):
        observed = metric_value(metric, target, second_score) - metric_value(
            metric,
            target,
            first_score,
        )
        bootstrap = []
        for _ in range(BOOTSTRAP_SAMPLES):
            group_weights = rng.multinomial(
                len(students),
                np.repeat(1 / len(students), len(students)),
            )
            weights = group_weights[group_codes]
            if (
                weights[target == 0].sum() == 0
                or weights[target == 1].sum() == 0
            ):
                continue
            bootstrap.append(
                metric_value(metric, target, second_score, weights)
                - metric_value(metric, target, first_score, weights)
            )

        extreme = 0
        for _ in range(PERMUTATION_SAMPLES):
            swap_student = rng.integers(
                0,
                2,
                len(students),
                dtype=np.int8,
            ).astype(bool)
            swap = swap_student[group_codes]
            permuted_first = np.where(swap, second_score, first_score)
            permuted_second = np.where(swap, first_score, second_score)
            difference = metric_value(
                metric,
                target,
                permuted_second,
            ) - metric_value(metric, target, permuted_first)
            if abs(difference) >= abs(observed):
                extreme += 1

        lower, upper = np.percentile(bootstrap, [2.5, 97.5])
        p_value = (extreme + 1) / (PERMUTATION_SAMPLES + 1)
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "difference": observed,
                "ci_low": lower,
                "ci_high": upper,
                "permutation_p": p_value,
                "significant_0_05": p_value < 0.05,
                "test_rows": len(paired),
                "test_students": len(students),
            }
        )
    return rows


def run_paired_tests(predictions):
    xgb = predictions.loc[predictions["model"].eq("XGBoost")]
    comparisons = []
    for institution, days, source in (
        ("UCI_dataset", [0], "uci_source"),
        ("OULAD_dataset", SNAPSHOT_DAYS, "oulad_source"),
    ):
        for day in days:
            first = xgb.loc[
                xgb["training_scenario"].eq(source)
                & xgb["institution"].eq(institution)
                & xgb["snapshot_day"].eq(day)
            ]
            second = xgb.loc[
                xgb["training_scenario"].eq("pooled")
                & xgb["institution"].eq(institution)
                & xgb["snapshot_day"].eq(day)
            ]
            label = f"{institution} day {day}: pooled minus local-only"
            comparisons.append((first, second, label))

    rows = []
    for index, (first, second, label) in enumerate(comparisons):
        rows.extend(
            paired_cluster_test(
                first,
                second,
                label,
                RANDOM_STATE + 100 + index,
            )
        )
    return pd.DataFrame(rows)


def calculate_dynamic_shap(model, uci, oulad):
    test = pd.concat(
        [
            uci.loc[uci["split"].eq("test")],
            oulad.loc[oulad["split"].eq("test")],
        ],
        ignore_index=True,
    )
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["model"]
    explainer = shap.TreeExplainer(classifier)
    rows = []

    for (institution, day), group in test.groupby(
        ["institution", "snapshot_day"]
    ):
        sample = group.sample(
            n=min(2000, len(group)),
            random_state=RANDOM_STATE,
        )
        transformed = imputer.transform(sample[MODEL_FEATURES])
        values = np.asarray(explainer.shap_values(transformed))
        if values.ndim == 3:
            mean_absolute = np.abs(values).mean(axis=(0, 2))
        else:
            mean_absolute = np.abs(values).mean(axis=0)
        for feature, value in zip(MODEL_FEATURES, mean_absolute):
            rows.append(
                {
                    "institution": institution,
                    "snapshot_day": int(day),
                    "feature": feature,
                    "mean_absolute_shap": float(value),
                    "sample_rows": len(sample),
                }
            )
    return pd.DataFrame(rows)


def plot_dynamic_shap(summary):
    labels = {
        ("UCI_dataset", 0): "UCI enrolment",
        ("OULAD_dataset", 0): "OULAD enrolment",
        ("OULAD_dataset", 35): "OULAD day 35",
        ("OULAD_dataset", 60): "OULAD day 60",
        ("OULAD_dataset", 75): "OULAD day 75",
    }
    plot = summary.copy()
    plot["context"] = [
        labels[(institution, int(day))]
        for institution, day in zip(
            plot["institution"],
            plot["snapshot_day"],
        )
    ]
    pivot = plot.pivot(
        index="feature",
        columns="context",
        values="mean_absolute_shap",
    )
    top = pivot.mean(axis=1).nlargest(8).index
    pivot = pivot.loc[top].sort_values("OULAD day 35")
    axis = pivot.plot.barh(
        figsize=(9.2, 6.0),
        color=["#D9822B", "#A9C6C2", "#4F8F87", "#2E6F68", "#184E4A"],
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("")
    axis.set_title("One portable dynamic model across five test contexts")
    axis.legend(title="Held-out context", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        FIGURES / "dynamic_unified_shap.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def compare_with_oulad_full_model(metrics):
    new = metrics.loc[
        metrics["model"].eq("XGBoost")
        & metrics["training_scenario"].eq("pooled")
        & metrics["test_institution"].eq("OULAD_dataset")
        & metrics["snapshot_day"].isin([35, 60, 75])
    ][
        [
            "snapshot_day",
            "roc_auc",
            "average_precision",
            "f1",
            "precision",
            "recall",
        ]
    ].copy()
    new["model_scope"] = "portable_dynamic_16"

    old_path = RESULTS / "journal_model_metrics.csv"
    if not old_path.exists():
        return new
    old = pd.read_csv(old_path)
    old = old.loc[
        old["cohort"].eq("dynamic_landmark")
        & old["model"].eq("XGBoost")
        & old["cutoff_day"].isin([35, 60, 75])
    ][
        [
            "cutoff_day",
            "roc_auc",
            "average_precision",
            "f1",
            "precision",
            "recall",
        ]
    ].rename(columns={"cutoff_day": "snapshot_day"})
    old["model_scope"] = "oulad_specific_51"
    return pd.concat([new, old], ignore_index=True)


def plot_performance(metrics):
    selected = metrics.loc[
        metrics["model"].eq("XGBoost")
        & metrics["training_scenario"].eq("pooled")
    ].copy()
    selected["test"] = np.where(
        selected["test_institution"].eq("UCI_dataset"),
        "UCI enrolment",
        "OULAD day " + selected["snapshot_day"].astype(str),
    )
    plot = selected.set_index("test")[["roc_auc", "average_precision", "f1"]]
    axis = plot.plot.bar(
        figsize=(9.0, 5.4),
        color=["#0F4C5C", "#E6A33D", "#7A5195"],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_title("Portable 16-feature pooled dynamic model")
    axis.legend(["ROC-AUC", "Average precision", "F1"])
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        FIGURES / "dynamic_unified_performance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


def save_artifact(model, threshold, training):
    MODELS.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model,
        "schema_version": SCHEMA_VERSION,
        "features": MODEL_FEATURES,
        "core_features": CORE_FEATURES,
        "optional_features": OPTIONAL_FEATURES,
        "feature_groups": {
            "background": BACKGROUND_FEATURES,
            "assessment": ASSESSMENT_FEATURES,
            "activity": ACTIVITY_FEATURES,
        },
        "validation_threshold": threshold,
        "score_semantics": (
            "Uncalibrated ranking score; use within-course or within-cohort "
            "percentiles for review."
        ),
        "target_definition": "future attrition after the snapshot",
        "training_rows": len(training),
        "training_students": training["student_id"].nunique(),
        "random_state": RANDOM_STATE,
    }
    joblib.dump(artifact, MODEL_PATH)

    contract = {
        key: value
        for key, value in artifact.items()
        if key != "pipeline"
    }
    contract["feature_schema"] = SCHEMA_ROWS
    contract["inference_policy"] = {
        "optional_missing_values": (
            "Median imputation; structured optional groups were masked "
            "during training so missing feeds do not change the schema."
        ),
        "required_missing_values": "Reject the record.",
        "recommended_output": "Within-course or within-cohort percentile",
        "external_validation_required": True,
        "school_specific_retraining_required": False,
    }
    CONTRACT_PATH.write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )


def check_split_overlap(frame):
    sets = {
        name: set(group["student_id"])
        for name, group in frame.groupby("split")
    }
    overlap = {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }
    if max(overlap.values()) != 0:
        raise ValueError(f"Student split overlap remains: {overlap}")
    return overlap


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Step 1: Build UCI enrolment snapshots")
    uci = make_uci_snapshots()
    validate_frame(uci)

    print("Step 2: Build OULAD day 0/35/60/75 snapshots")
    oulad = make_oulad_snapshots()
    validate_frame(oulad)

    print("Step 3: Assign student-exclusive splits")
    uci = assign_splits(uci, RANDOM_STATE)
    oulad = assign_splits(oulad, RANDOM_STATE + 1)
    split_rows = []
    for institution, frame in (
        ("UCI_dataset", uci),
        ("OULAD_dataset", oulad),
    ):
        overlap = check_split_overlap(frame)
        overlap["institution"] = institution
        split_rows.append(overlap)
    pd.DataFrame(split_rows).to_csv(
        RESULTS / "dynamic_unified_split_overlap.csv",
        index=False,
    )

    all_rows = pd.concat([uci, oulad], ignore_index=True)
    (
        all_rows.groupby(
            ["institution", "snapshot_day", "split"],
            as_index=False,
        )
        .agg(
            rows=("record_id", "size"),
            students=("student_id", "nunique"),
            prevalence=("target", "mean"),
        )
        .to_csv(
            RESULTS / "dynamic_unified_cohort_summary.csv",
            index=False,
        )
    )
    pd.DataFrame(SCHEMA_ROWS).to_csv(
        RESULTS / "dynamic_unified_feature_schema.csv",
        index=False,
    )
    availability_summary(all_rows).to_csv(
        RESULTS / "dynamic_unified_feature_availability.csv",
        index=False,
    )

    print("Step 4: Fit local, zero-shot, and pooled models")
    (
        metrics,
        predictions,
        thresholds,
        final_model,
        final_threshold,
    ) = run_experiments(uci, oulad)
    metrics.to_csv(
        RESULTS / "dynamic_unified_model_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        RESULTS / "dynamic_unified_model_predictions.csv",
        index=False,
    )
    thresholds.to_csv(
        RESULTS / "dynamic_unified_threshold_selection.csv",
        index=False,
    )
    compare_with_oulad_full_model(metrics).to_csv(
        RESULTS / "dynamic_unified_model_comparison.csv",
        index=False,
    )
    run_paired_tests(predictions).to_csv(
        RESULTS / "dynamic_unified_significance.csv",
        index=False,
    )
    feature_importance(final_model).to_csv(
        RESULTS / "dynamic_unified_feature_importance.csv",
        index=False,
    )
    shap_summary = calculate_dynamic_shap(final_model, uci, oulad)
    shap_summary.to_csv(
        RESULTS / "dynamic_unified_shap.csv",
        index=False,
    )
    plot_dynamic_shap(shap_summary)
    plot_performance(metrics)

    pooled_training = pd.concat(
        [
            uci.loc[uci["split"].eq("train")],
            oulad.loc[oulad["split"].eq("train")],
        ],
        ignore_index=True,
    )
    save_artifact(final_model, final_threshold, pooled_training)
    print("Saved", MODEL_PATH)
    print("Saved", CONTRACT_PATH)
    print(
        metrics.loc[
            metrics["model"].eq("XGBoost")
            & metrics["training_scenario"].eq("pooled")
        ][
            [
                "test_institution",
                "snapshot_day",
                "roc_auc",
                "average_precision",
                "f1",
                "precision",
                "recall",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
