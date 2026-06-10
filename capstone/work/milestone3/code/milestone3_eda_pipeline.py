from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
MILESTONE_DIR = SCRIPT_DIR.parent
RAW = MILESTONE_DIR / "data" / "raw"
FIG = MILESTONE_DIR / "data" / "figures"
OUT = MILESTONE_DIR / "data" / "analysis"

UCI_PATH = RAW / "uci_student_dropout.csv"
OULAD_DIR = RAW / "oulad"

sns.set_theme(style="whitegrid")


def savefig(name: str):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return str(path)


def iqr_outlier_count(series: pd.Series) -> int:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def run_uci():
    df = pd.read_csv(UCI_PATH)
    original_shape = list(df.shape)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df["is_dropout"] = (df["Target"] == "Dropout").astype(int)

    target_counts = df["Target"].value_counts().to_dict()
    missing_total = int(df.isna().sum().sum())
    duplicate_total = int(df.duplicated().sum())

    selected_cols = [
        "Admission grade",
        "Previous qualification (grade)",
        "Age at enrollment",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
    ]
    selected_cols = [c for c in selected_cols if c in df.columns]

    group_means = (
        df.groupby("Target")[selected_cols]
        .mean(numeric_only=True)
        .round(2)
        .to_dict(orient="index")
    )

    corr_candidates = [
        "Admission grade",
        "Previous qualification (grade)",
        "Age at enrollment",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Unemployment rate",
        "Inflation rate",
        "GDP",
    ]
    corr_candidates = [c for c in corr_candidates if c in df.columns]
    corr = df[corr_candidates].corr().round(2)

    # Correlation of selected features with dropout binary
    dropout_corr = (
        df[corr_candidates + ["is_dropout"]]
        .corr()["is_dropout"]
        .drop("is_dropout")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .round(3)
        .to_dict()
    )

    outlier_counts = {
        col: iqr_outlier_count(df[col])
        for col in [
            "Admission grade",
            "Previous qualification (grade)",
            "Age at enrollment",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
        ]
        if col in df.columns
    }

    # Figures
    plt.figure(figsize=(7, 4))
    order = ["Graduate", "Dropout", "Enrolled"]
    sns.countplot(data=df, x="Target", order=order, palette="Set2")
    plt.title("UCI Target Distribution")
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    uci_target_fig = savefig("uci_target_distribution.png")

    if "Curricular units 1st sem (approved)" in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(
            data=df,
            x="Target",
            y="Curricular units 1st sem (approved)",
            order=order,
            palette="Set2",
        )
        plt.title("UCI First-Semester Approved Units by Outcome")
        plt.xlabel("Outcome")
        plt.ylabel("Approved Units")
        uci_box_fig = savefig("uci_first_sem_approved_by_target.png")
    else:
        uci_box_fig = None

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
    plt.title("UCI Correlation Heatmap (Selected Numeric Variables)")
    uci_corr_fig = savefig("uci_selected_correlation_heatmap.png")

    return {
        "shape": original_shape,
        "missing_total": missing_total,
        "duplicate_total": duplicate_total,
        "numeric_feature_count": len(numeric_cols),
        "target_counts": target_counts,
        "group_means": group_means,
        "dropout_corr": dropout_corr,
        "outlier_counts": outlier_counts,
        "figures": {
            "target_distribution": uci_target_fig,
            "approved_units_boxplot": uci_box_fig,
            "correlation_heatmap": uci_corr_fig,
        },
    }


def aggregate_student_virtual_learning_environment(path: Path) -> pd.DataFrame:
    aggregate = None
    for chunk in pd.read_csv(path, chunksize=500_000):
        grouped = (
            chunk.groupby(["code_module", "code_presentation", "id_student"])
            .agg(
                virtual_learning_environment_event_count=("sum_click", "size"),
                total_clicks=("sum_click", "sum"),
            )
            .reset_index()
        )
        if aggregate is None:
            aggregate = grouped
        else:
            aggregate = pd.concat([aggregate, grouped], ignore_index=True)
            aggregate = (
                aggregate.groupby(["code_module", "code_presentation", "id_student"], as_index=False)
                .sum()
            )
    return aggregate


def run_oulad():
    student_info = pd.read_csv(OULAD_DIR / "studentInfo.csv")
    student_registration = pd.read_csv(OULAD_DIR / "studentRegistration.csv")
    student_assessment = pd.read_csv(OULAD_DIR / "studentAssessment.csv")
    assessments = pd.read_csv(OULAD_DIR / "assessments.csv")
    courses = pd.read_csv(OULAD_DIR / "courses.csv")
    virtual_learning_environment_metadata = pd.read_csv(OULAD_DIR / "vle.csv")
    student_virtual_learning_environment_agg = (
        aggregate_student_virtual_learning_environment(OULAD_DIR / "studentVle.csv")
    )

    table_shapes = {
        "studentInfo": list(student_info.shape),
        "studentRegistration": list(student_registration.shape),
        "studentAssessment": list(student_assessment.shape),
        "assessments": list(assessments.shape),
        "courses": list(courses.shape),
        "virtual_learning_environment_metadata": list(
            virtual_learning_environment_metadata.shape
        ),
        "student_virtual_learning_environment_aggregated_student_level": list(
            student_virtual_learning_environment_agg.shape
        ),
    }
    table_missing = {
        "studentInfo": student_info.isna().sum().to_dict(),
        "studentRegistration": student_registration.isna().sum().to_dict(),
        "studentAssessment": student_assessment.isna().sum().to_dict(),
    }
    table_duplicates = {
        "studentInfo": int(student_info.duplicated().sum()),
        "studentRegistration": int(student_registration.duplicated().sum()),
        "studentAssessment": int(student_assessment.duplicated().sum()),
    }

    target_counts = student_info["final_result"].value_counts().to_dict()

    key_numeric = ["studied_credits", "num_of_prev_attempts"]
    info_group_means = (
        student_info.groupby("final_result")[key_numeric]
        .mean(numeric_only=True)
        .round(2)
        .to_dict(orient="index")
    )

    # Assessment aggregates
    assess_agg = (
        student_assessment.groupby(["id_assessment", "id_student"])
        .agg(score=("score", "mean"), date_submitted=("date_submitted", "mean"))
        .reset_index()
    )
    assess_agg = assess_agg.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "assessment_type", "date", "weight"]],
        on="id_assessment",
        how="left",
    )
    student_assess_agg = (
        assess_agg.groupby(["code_module", "code_presentation", "id_student"])
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            assessment_count=("score", "size"),
            mean_weight=("weight", "mean"),
        )
        .reset_index()
    )

    merged = student_info.merge(
        student_registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    merged = merged.merge(
        student_assess_agg,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )
    merged = merged.merge(
        student_virtual_learning_environment_agg,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    merged["is_withdrawn"] = (merged["final_result"] == "Withdrawn").astype(int)
    merged["registration_length_days"] = (
        merged["date_unregistration"] - merged["date_registration"]
    )
    merged["registration_length_days"] = merged["registration_length_days"].where(
        merged["date_unregistration"].notna()
    )

    # Handle absent aggregated behavior as zeros for descriptive purposes
    for col in [
        "assessment_count",
        "mean_score",
        "virtual_learning_environment_event_count",
        "total_clicks",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    oulad_group_means = (
        merged.groupby("final_result")[
            [
                "studied_credits",
                "num_of_prev_attempts",
                "assessment_count",
                "mean_score",
                "virtual_learning_environment_event_count",
                "total_clicks",
            ]
        ]
        .mean(numeric_only=True)
        .round(2)
        .to_dict(orient="index")
    )

    # Figures
    order = ["Pass", "Withdrawn", "Fail", "Distinction"]
    order = [x for x in order if x in merged["final_result"].unique()]

    plt.figure(figsize=(7, 4))
    sns.countplot(data=merged, x="final_result", order=order, palette="Set2")
    plt.title("OULAD Final Result Distribution")
    plt.xlabel("Final Result")
    plt.ylabel("Count")
    oulad_target_fig = savefig("oulad_final_result_distribution.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=merged, x="final_result", y="studied_credits", order=order, palette="Set2")
    plt.title("OULAD Studied Credits by Final Result")
    plt.xlabel("Final Result")
    plt.ylabel("Studied Credits")
    oulad_credits_fig = savefig("oulad_studied_credits_by_result.png")

    plot_df = merged.copy()
    plot_df["log_total_clicks"] = np.log1p(plot_df["total_clicks"])
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=plot_df, x="final_result", y="log_total_clicks", order=order, palette="Set2")
    plt.title("OULAD Log Total Virtual Learning Environment Clicks by Final Result")
    plt.xlabel("Final Result")
    plt.ylabel("log(1 + total clicks)")
    oulad_clicks_fig = savefig("oulad_log_clicks_by_result.png")

    plt.figure(figsize=(7, 4))
    sns.boxplot(data=merged, x="final_result", y="mean_score", order=order, palette="Set2")
    plt.title("OULAD Mean Assessment Score by Final Result")
    plt.xlabel("Final Result")
    plt.ylabel("Mean Assessment Score")
    oulad_score_fig = savefig("oulad_mean_score_by_result.png")

    return {
        "table_shapes": table_shapes,
        "table_missing": table_missing,
        "table_duplicates": table_duplicates,
        "target_counts": target_counts,
        "student_info_group_means": info_group_means,
        "merged_group_means": oulad_group_means,
        "registration_missing_date_unregistration": int(merged["date_unregistration"].isna().sum()),
        "registration_nonmissing_date_unregistration": int(merged["date_unregistration"].notna().sum()),
        "figures": {
            "target_distribution": oulad_target_fig,
            "studied_credits_boxplot": oulad_credits_fig,
            "log_clicks_boxplot": oulad_clicks_fig,
            "mean_score_boxplot": oulad_score_fig,
        },
    }


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    uci = run_uci()
    oulad = run_oulad()
    result = {"uci": uci, "oulad": oulad}
    out_path = OUT / "milestone3_eda_summary.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
