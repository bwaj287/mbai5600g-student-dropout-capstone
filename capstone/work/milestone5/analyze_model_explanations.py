import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
MILESTONE4 = ROOT.parent / "milestone4"
SCHEMA_PATH = MILESTONE4 / "data" / "shared_feature_schema.json"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SHAP_FILES = {
    "uci_multiclass_xgboost": RESULTS / "uci_multiclass_xgboost_shap_summary.csv",
    "uci_binary_early_xgboost": RESULTS / "uci_binary_early_xgboost_shap_summary.csv",
    "oulad_binary_early_xgboost": RESULTS / "oulad_binary_early_xgboost_shap_summary.csv",
}

TASK_LABELS = {
    "uci_binary_early_xgboost": "UCI early XGBoost",
    "oulad_binary_early_xgboost": "OULAD early XGBoost",
}

FEATURE_FAMILY_LABELS = {
    "academic_progress": "Academic progress",
    "early_academic_progress": "Early academic progress",
    "early_engagement": "Early engagement",
    "financial_support": "Financial support",
    "program_setup": "Program setup",
    "demographics": "Demographics",
    "prior_preparation": "Prior preparation",
    "macro_context": "Macro context",
    "other": "Other",
}


def rel(path):
    return str(path.relative_to(REPO_ROOT))


def clean_feature_name(feature):
    for prefix in ("numeric__", "categorical__"):
        if feature.startswith(prefix):
            return feature.replace(prefix, "", 1)
    return feature


def build_family_lookup(dataset_key):
    schema = json.loads(SCHEMA_PATH.read_text())
    lookup = {}
    for family, mappings in schema["feature_families"].items():
        for feature in mappings.get(dataset_key, []):
            lookup[feature] = family
    return lookup


def classify_feature(feature, dataset_key, lookup):
    cleaned = clean_feature_name(feature)
    matches = sorted(
        (name for name in lookup if cleaned == name or cleaned.startswith(f"{name}_")),
        key=len,
        reverse=True,
    )
    if matches:
        return lookup[matches[0]]

    if dataset_key == "oulad":
        if cleaned.startswith("vle_"):
            return "early_engagement"
        if cleaned.startswith("assessment_") or cleaned.startswith("early_assessment"):
            return "early_academic_progress"
        if cleaned.startswith("code_module") or cleaned.startswith("code_presentation"):
            return "program_setup"

    if dataset_key == "uci":
        if cleaned.startswith("Curricular units"):
            return "academic_progress"
        if cleaned.startswith("Course") or cleaned.startswith("Application"):
            return "program_setup"
        if cleaned.startswith("Debtor") or cleaned.startswith("Tuition") or cleaned.startswith("Scholarship"):
            return "financial_support"

    return "other"


def summarize_one(task_model, shap_path):
    dataset_key = "oulad" if task_model.startswith("oulad") else "uci"
    lookup = build_family_lookup(dataset_key)
    frame = pd.read_csv(shap_path)
    frame["feature_family"] = [
        classify_feature(feature, dataset_key, lookup) for feature in frame["feature"]
    ]
    summary = (
        frame.groupby("feature_family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    total = summary["mean_abs_shap"].sum()
    summary["share_of_total_shap"] = summary["mean_abs_shap"] / total if total else 0
    summary["task_model"] = task_model
    return summary[["task_model", "feature_family", "mean_abs_shap", "share_of_total_shap"]]


def save_family_plot(family_frame):
    plot_tasks = ["uci_binary_early_xgboost", "oulad_binary_early_xgboost"]
    plot_frame = family_frame[family_frame["task_model"].isin(plot_tasks)].copy()
    plot_frame["task_label"] = plot_frame["task_model"].map(TASK_LABELS)
    plot_frame["feature_family_label"] = plot_frame["feature_family"].map(FEATURE_FAMILY_LABELS)
    pivot = plot_frame.pivot_table(
        index="feature_family_label",
        columns="task_label",
        values="share_of_total_shap",
        fill_value=0,
    )
    plot_columns = ["UCI early XGBoost", "OULAD early XGBoost"]
    pivot = pivot.reindex(columns=plot_columns).fillna(0)
    pivot = pivot.sort_values(plot_columns, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    pivot.plot(kind="barh", ax=ax)
    ax.set_title("SHAP Feature-Family Comparison: UCI vs OULAD Early Warning")
    ax.set_xlabel("Share of total mean absolute SHAP")
    ax.set_ylabel("Feature family")
    ax.legend(plot_columns, loc="best")
    fig.tight_layout()
    out_path = FIGURES / "shap_feature_family_comparison.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rel(out_path)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    frames = []
    for task_model, shap_path in SHAP_FILES.items():
        if shap_path.exists():
            frames.append(summarize_one(task_model, shap_path))

    family_frame = pd.concat(frames, ignore_index=True)
    out_path = RESULTS / "shap_feature_family_comparison.csv"
    family_frame.to_csv(out_path, index=False)
    plot_path = save_family_plot(family_frame)

    payload = {
        "feature_family_summary": rel(out_path),
        "feature_family_plot": plot_path,
        "interpretation": {
            "uci": "UCI explanations are concentrated in academic progress, financial support, and demographic/program variables.",
            "oulad": "OULAD explanations are concentrated in early assessment behavior, engagement, studied credits, and module/presentation context.",
            "project_relevance": "The shifted explanation patterns show how model explanations differ across two educational settings.",
        },
    }
    summary_path = RESULTS / "explanation_shift_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
