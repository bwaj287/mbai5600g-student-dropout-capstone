from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CAPSTONE = ROOT.parents[1]
REPO_ROOT = CAPSTONE.parent
OUT_DATA = ROOT / "data"
OUT_RESULTS = ROOT / "results"

EARLY_CUTOFF_DAYS = 75
RANDOM_STATE = 42

UCI_NUMERIC_COLUMNS = [
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
]

UCI_SECOND_SEMESTER_COLUMNS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

SHARED_FEATURE_SCHEMA = {
    "targets": {
        "uci_multiclass": "Target",
        "uci_binary_attrition": "is_attrition",
        "oulad_binary_attrition": "is_attrition",
    },
    "feature_families": {
        "demographics": {
            "uci": [
                "Marital Status",
                "Nacionality",
                "Gender",
                "Age at enrollment",
                "International",
            ],
            "oulad": [
                "gender",
                "region",
                "age_band",
                "imd_band",
                "disability",
            ],
        },
        "prior_preparation": {
            "uci": [
                "Previous qualification",
                "Previous qualification (grade)",
                "Mother's qualification",
                "Father's qualification",
            ],
            "oulad": [
                "highest_education",
                "num_of_prev_attempts",
            ],
        },
        "program_setup": {
            "uci": [
                "Course",
                "Application mode",
                "Application order",
                "Daytime/evening attendance",
            ],
            "oulad": [
                "code_module",
                "code_presentation",
                "studied_credits",
                "module_presentation_length",
                "date_registration",
            ],
        },
        "financial_support": {
            "uci": [
                "Debtor",
                "Tuition fees up to date",
                "Scholarship holder",
            ],
            "oulad": [],
        },
        "early_academic_progress": {
            "uci": [
                "Curricular units 1st sem (enrolled)",
                "Curricular units 1st sem (evaluations)",
                "Curricular units 1st sem (approved)",
                "Curricular units 1st sem (grade)",
            ],
            "oulad": [
                "assessment_submission_count_early",
                "assessment_submission_ratio_early",
                "assessment_score_mean_early",
                "assessment_weighted_score_ratio_early",
            ],
        },
        "early_engagement": {
            "uci": [],
            "oulad": [
                "vle_total_clicks_early",
                "vle_event_count_early",
                "vle_active_days_early",
                "vle_unique_sites_early",
            ],
        },
        "macro_context": {
            "uci": [
                "Unemployment rate",
                "Inflation rate",
                "GDP",
            ],
            "oulad": [],
        },
    },
    "notes": {
        "cutoff_days": EARLY_CUTOFF_DAYS,
        "rationale": (
            "Use a fixed 75-day early-course cutoff for OULAD so every module/presentation "
            "retains at least one scheduled assessment while remaining before the midpoint "
            "for most presentations."
        ),
        "label_mapping": {
            "uci": {
                "attrition_positive": ["Dropout"],
                "attrition_negative": ["Graduate", "Enrolled"],
            },
            "oulad": {
                "attrition_positive": ["Withdrawn"],
                "attrition_negative": ["Pass", "Distinction", "Fail"],
            },
        },
    },
}


def ensure_dirs() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def summarize_missingness(frame: pd.DataFrame) -> dict[str, float]:
    missing = (frame.isna().mean() * 100).round(2)
    missing = missing[missing > 0]
    return {column: float(value) for column, value in missing.sort_values(ascending=False).items()}


def find_raw_dir() -> Path:
    candidates = [
        CAPSTONE / "data" / "raw",
        CAPSTONE / "work" / "milestone3" / "data" / "raw",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find the project raw-data directory.")


def build_uci_datasets() -> dict[str, object]:
    raw_dir = find_raw_dir()
    df = pd.read_csv(raw_dir / "uci_student_dropout.csv")
    df["is_attrition"] = (df["Target"] == "Dropout").astype(int)

    all_columns = df.columns.tolist()
    categorical_columns = [
        column
        for column in all_columns
        if column not in UCI_NUMERIC_COLUMNS and column not in {"Target", "is_attrition"}
    ]

    multiclass_out = OUT_DATA / "uci_multiclass_model_ready.csv"
    df.to_csv(multiclass_out, index=False)

    early_warning_df = df.drop(columns=UCI_SECOND_SEMESTER_COLUMNS).copy()
    early_warning_out = OUT_DATA / "uci_binary_early_model_ready.csv"
    early_warning_df.to_csv(early_warning_out, index=False)

    return {
        "paths": {
            "multiclass": rel(multiclass_out),
            "binary_early": rel(early_warning_out),
        },
        "shape": list(df.shape),
        "target_distribution": df["Target"].value_counts().to_dict(),
        "binary_target_distribution": df["is_attrition"].value_counts().to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_column_pct": summarize_missingness(df),
        "duplicate_total": int(df.duplicated().sum()),
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(UCI_NUMERIC_COLUMNS),
        "early_warning_columns_removed": UCI_SECOND_SEMESTER_COLUMNS,
    }


def build_oulad_dataset() -> dict[str, object]:
    raw_dir = find_raw_dir()
    oulad_dir = raw_dir / "oulad"
    student_info = pd.read_csv(oulad_dir / "studentInfo.csv")
    student_registration = pd.read_csv(oulad_dir / "studentRegistration.csv")
    courses = pd.read_csv(oulad_dir / "courses.csv")
    assessments = pd.read_csv(oulad_dir / "assessments.csv")
    student_assessment = pd.read_csv(oulad_dir / "studentAssessment.csv")
    vle = pd.read_csv(oulad_dir / "vle.csv")
    student_vle = pd.read_csv(
        oulad_dir / "studentVle.csv",
        dtype={
            "code_module": "string",
            "code_presentation": "string",
            "id_student": "int32",
            "id_site": "int32",
            "date": "int16",
            "sum_click": "int32",
        },
    )

    student_info["is_attrition"] = (student_info["final_result"] == "Withdrawn").astype(int)

    early_assessments = assessments.loc[
        assessments["date"].fillna(np.inf) <= EARLY_CUTOFF_DAYS
    ].copy()
    early_assessment_base = student_assessment.merge(
        early_assessments,
        on="id_assessment",
        how="inner",
        suffixes=("", "_scheduled"),
    )
    early_assessment_base = early_assessment_base.loc[
        early_assessment_base["date_submitted"].fillna(np.inf) <= EARLY_CUTOFF_DAYS
    ].copy()
    early_assessment_base["weighted_score"] = (
        early_assessment_base["score"].fillna(0.0)
        * early_assessment_base["weight"].fillna(0.0)
        / 100.0
    )
    early_assessment_base["submission_delay_days"] = (
        early_assessment_base["date_submitted"] - early_assessment_base["date"]
    )
    early_assessment_base["late_submission"] = (
        early_assessment_base["submission_delay_days"] > 0
    ).astype(int)

    assessment_core = (
        early_assessment_base.groupby(
            ["code_module", "code_presentation", "id_student"], as_index=False
        )
        .agg(
            assessment_submission_count_early=("score", "size"),
            assessment_score_mean_early=("score", "mean"),
            assessment_score_std_early=("score", "std"),
            assessment_score_max_early=("score", "max"),
            assessment_score_min_early=("score", "min"),
            assessment_weighted_score_sum_early=("weighted_score", "sum"),
            assessment_mean_submission_delay_early=("submission_delay_days", "mean"),
            assessment_late_submission_count_early=("late_submission", "sum"),
            assessment_banked_count_early=("is_banked", "sum"),
        )
    )

    assessment_type_pivot = (
        early_assessment_base.pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="assessment_type",
            values="score",
            aggfunc="size",
            fill_value=0,
        )
        .rename(columns=lambda value: f"assessment_type_count_{str(value).lower()}_early")
        .reset_index()
    )

    assessment_schedule = (
        early_assessments.groupby(["code_module", "code_presentation"], as_index=False)
        .agg(
            early_assessment_count_expected=("id_assessment", "nunique"),
            early_assessment_weight_expected=("weight", "sum"),
        )
    )

    student_vle = student_vle.loc[student_vle["date"] <= EARLY_CUTOFF_DAYS].copy()
    vle_lookup = vle[
        ["id_site", "code_module", "code_presentation", "activity_type"]
    ].drop_duplicates()
    student_vle = student_vle.merge(
        vle_lookup,
        on=["id_site", "code_module", "code_presentation"],
        how="left",
    )

    vle_core = (
        student_vle.groupby(["code_module", "code_presentation", "id_student"], as_index=False)
        .agg(
            vle_total_clicks_early=("sum_click", "sum"),
            vle_event_count_early=("sum_click", "size"),
            vle_active_days_early=("date", "nunique"),
            vle_unique_sites_early=("id_site", "nunique"),
        )
    )

    vle_activity = (
        student_vle.pivot_table(
            index=["code_module", "code_presentation", "id_student"],
            columns="activity_type",
            values="sum_click",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns=lambda value: f"vle_clicks_{str(value).lower()}_early")
        .reset_index()
    )

    model_ready = student_info.merge(
        student_registration.drop(columns=["date_unregistration"]),
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    model_ready = model_ready.merge(
        courses, on=["code_module", "code_presentation"], how="left"
    )
    model_ready = model_ready.merge(
        assessment_schedule, on=["code_module", "code_presentation"], how="left"
    )
    model_ready = model_ready.merge(
        assessment_core,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    model_ready = model_ready.merge(
        assessment_type_pivot,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    model_ready = model_ready.merge(
        vle_core,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    model_ready = model_ready.merge(
        vle_activity,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    aggregate_columns = [
        column
        for column in model_ready.columns
        if column.endswith("_early")
        or column in {
            "early_assessment_count_expected",
            "early_assessment_weight_expected",
        }
    ]
    model_ready[aggregate_columns] = model_ready[aggregate_columns].fillna(0)
    model_ready["assessment_score_std_early"] = model_ready["assessment_score_std_early"].fillna(0)
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
    model_ready["registration_lead_days"] = (-model_ready["date_registration"]).clip(lower=0)

    out_path = OUT_DATA / "oulad_binary_early_model_ready.csv"
    model_ready.to_csv(out_path, index=False)

    return {
        "path": rel(out_path),
        "shape": list(model_ready.shape),
        "target_distribution": student_info["final_result"].value_counts().to_dict(),
        "binary_target_distribution": model_ready["is_attrition"].value_counts().to_dict(),
        "missing_total": int(model_ready.isna().sum().sum()),
        "missing_by_column_pct": summarize_missingness(model_ready),
        "duplicate_total": int(model_ready.duplicated().sum()),
        "module_presentations": int(
            model_ready[["code_module", "code_presentation"]].drop_duplicates().shape[0]
        ),
        "cutoff_days": EARLY_CUTOFF_DAYS,
        "aggregate_feature_count": len(aggregate_columns) + 3,
        "vle_activity_feature_count": int(
            len([column for column in model_ready.columns if column.startswith("vle_clicks_")])
        ),
        "assessment_type_feature_count": int(
            len(
                [
                    column
                    for column in model_ready.columns
                    if column.startswith("assessment_type_count_")
                ]
            )
        ),
        "retained_missing_for_imputation": [
            column for column, pct in summarize_missingness(model_ready).items() if pct > 0
        ],
    }


def write_schema_files() -> dict[str, str]:
    schema_json_path = OUT_DATA / "shared_feature_schema.json"
    schema_json_path.write_text(json.dumps(SHARED_FEATURE_SCHEMA, indent=2))

    rows = []
    for family, mapping in SHARED_FEATURE_SCHEMA["feature_families"].items():
        for dataset_name in ("uci", "oulad"):
            for feature_name in mapping[dataset_name]:
                rows.append(
                    {
                        "family": family,
                        "dataset": dataset_name,
                        "feature_name": feature_name,
                    }
                )
    schema_csv_path = OUT_DATA / "shared_feature_schema.csv"
    pd.DataFrame(rows).to_csv(schema_csv_path, index=False)

    return {"json": rel(schema_json_path), "csv": rel(schema_csv_path)}


def main() -> None:
    ensure_dirs()
    uci_summary = build_uci_datasets()
    oulad_summary = build_oulad_dataset()
    schema_paths = write_schema_files()

    summary = {
        "random_state": RANDOM_STATE,
        "uci": uci_summary,
        "oulad": oulad_summary,
        "shared_schema_paths": schema_paths,
    }
    (OUT_RESULTS / "prep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
