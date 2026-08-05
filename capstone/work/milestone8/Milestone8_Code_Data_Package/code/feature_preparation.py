from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
RAW_OULAD = PACKAGE / "data" / "raw" / "oulad"


TIME_WINDOWS = [35, 60, 75]
KEY_COLUMNS = ["code_module", "code_presentation", "id_student"]


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
