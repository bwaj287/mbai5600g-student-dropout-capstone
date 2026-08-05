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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parent
RAW = PACKAGE / "data" / "raw"
RESULTS = PACKAGE / "outputs" / "results"
FIGURES = PACKAGE / "outputs" / "figures"
MODELS = PACKAGE / "models"

MODEL_PATH = MODELS / "unified_enrolment_xgboost.joblib"
CONTRACT_PATH = MODELS / "unified_enrolment_feature_contract.json"

RANDOM_STATE = 42
BOOTSTRAP_SAMPLES = 1000
PERMUTATION_SAMPLES = 1000
SCHEMA_VERSION = "2.0"

# These are deliberately limited to concepts available at enrolment in both
# source datasets. Assessment and LMS features remain in the separate OULAD
# landmark analysis because UCI has no equivalent dated event history.
CORE_FEATURES = [
    "age_scaled",
    "prior_preparation",
    "study_load",
]
OPTIONAL_FEATURES = [
    "male",
    "declared_support_need",
    "financial_stability",
]
ENROLMENT_FEATURES = CORE_FEATURES + OPTIONAL_FEATURES

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
        "shared_feature": "age_scaled",
        "required": True,
        "meaning": "Age position on a fixed 18-60 scale",
        "uci_mapping": "Age at enrollment",
        "oulad_mapping": "Midpoint of age_band",
        "new_school_requirement": "Age or compatible age band",
    },
    {
        "shared_feature": "prior_preparation",
        "required": True,
        "meaning": "Prior education and preparation score",
        "uci_mapping": "Previous qualification and grade",
        "oulad_mapping": "highest_education and prior attempts",
        "new_school_requirement": "Prior preparation mapped to 0-1",
    },
    {
        "shared_feature": "study_load",
        "required": True,
        "meaning": "Registered study load relative to a heavy load",
        "uci_mapping": "First-semester enrolled units / 12",
        "oulad_mapping": "studied_credits / 120",
        "new_school_requirement": "Registered load mapped to 0-1",
    },
    {
        "shared_feature": "male",
        "required": False,
        "meaning": "Optional documented binary gender indicator",
        "uci_mapping": "Gender",
        "oulad_mapping": "gender",
        "new_school_requirement": "0/1 if approved; otherwise blank",
    },
    {
        "shared_feature": "declared_support_need",
        "required": False,
        "meaning": "Optional declared disability or support need",
        "uci_mapping": "Educational special needs",
        "oulad_mapping": "disability",
        "new_school_requirement": "0/1 if approved; otherwise blank",
    },
    {
        "shared_feature": "financial_stability",
        "required": False,
        "meaning": "Optional documented financial-stability proxy",
        "uci_mapping": "Debtor and tuition-fee status",
        "oulad_mapping": "imd_band midpoint",
        "new_school_requirement": "Comparable 0-1 value or blank",
    },
]


def imd_midpoint(values):
    extracted = values.astype("string").str.extract(r"(\d+)-(\d+)%")
    low = pd.to_numeric(extracted[0], errors="coerce")
    high = pd.to_numeric(extracted[1], errors="coerce")
    return (low + high) / 200


def make_uci_adapter():
    raw = pd.read_csv(RAW / "uci_student_dropout.csv")
    frame = pd.DataFrame(index=raw.index)
    frame["institution"] = "UCI_dataset"
    frame["record_id"] = "UCI_" + raw.index.astype(str)
    frame["student_id"] = frame["record_id"]
    frame["target"] = raw["Target"].eq("Dropout").astype(int)

    frame["age_scaled"] = (
        (raw["Age at enrollment"] - 18) / 42
    ).clip(0, 1)
    education = raw["Previous qualification"].map(UCI_EDUCATION)
    prior_grade = (
        raw["Previous qualification (grade)"] / 200
    ).clip(0, 1)
    frame["prior_preparation"] = (
        0.5 * education.fillna(0.5)
        + 0.5 * prior_grade.fillna(0.5)
    )
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
    return frame


def make_oulad_adapter():
    # The portable model uses only enrolment-time studentInfo fields. It does
    # not load studentVle or assessment outcomes, so its target and observation
    # point are aligned with the UCI eventual-attrition task as closely as the
    # two public datasets permit.
    raw = pd.read_csv(RAW / "oulad" / "studentInfo.csv")
    frame = pd.DataFrame(index=raw.index)
    frame["institution"] = "OULAD_dataset"
    frame["record_id"] = (
        "OULAD_"
        + raw["code_module"].astype(str)
        + "_"
        + raw["code_presentation"].astype(str)
        + "_"
        + raw["id_student"].astype(str)
    )
    frame["student_id"] = "OULAD_" + raw["id_student"].astype(str)
    frame["target"] = raw["final_result"].eq("Withdrawn").astype(int)

    age_midpoint = raw["age_band"].map(
        {"0-35": 26.0, "35-55": 45.0, "55<=": 60.0}
    )
    frame["age_scaled"] = (
        (age_midpoint.fillna(35.0) - 18) / 42
    ).clip(0, 1)
    education = raw["highest_education"].map(OULAD_EDUCATION)
    prior_attempts = (raw["num_of_prev_attempts"] / 3).clip(0, 1)
    frame["prior_preparation"] = (
        0.8 * education.fillna(0.5)
        + 0.2 * (1 - prior_attempts.fillna(0.0))
    )
    frame["study_load"] = (raw["studied_credits"] / 120).clip(0, 1)
    frame["male"] = raw["gender"].map({"M": 1.0, "F": 0.0})
    frame["declared_support_need"] = raw["disability"].map(
        {"Y": 1.0, "N": 0.0}
    )
    frame["financial_stability"] = imd_midpoint(raw["imd_band"])
    return frame


def validate_adapter(frame, require_target=True):
    required = {
        "institution",
        "record_id",
        "student_id",
        *CORE_FEATURES,
    }
    if require_target:
        required.add("target")

    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Adapter is missing columns: {missing}")
    if frame["record_id"].duplicated().any():
        raise ValueError("record_id values must be unique.")
    if require_target and not frame["target"].isin([0, 1]).all():
        raise ValueError("The target column must contain only 0 and 1.")

    for feature in ENROLMENT_FEATURES:
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
    for feature in OPTIONAL_FEATURES:
        if feature not in output:
            output[feature] = np.nan
    for feature in ENROLMENT_FEATURES:
        if feature in output:
            output[feature] = pd.to_numeric(
                output[feature],
                errors="coerce",
            )
    validate_adapter(output, require_target=False)
    return output


def assign_splits(frame):
    target_by_student = frame.groupby("student_id")["target"].max()
    students = target_by_student.index.to_numpy()
    labels = target_by_student.to_numpy()
    train_adaptation, test = train_test_split(
        students,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    remaining_labels = target_by_student.loc[train_adaptation].to_numpy()
    train, adaptation = train_test_split(
        train_adaptation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=remaining_labels,
    )
    assignments = {student: "train" for student in train}
    assignments.update({student: "adaptation" for student in adaptation})
    assignments.update({student: "test" for student in test})
    output = frame.copy()
    output["split"] = output["student_id"].map(assignments)
    return output


def select_adaptation_rows(frame, percent):
    candidates = frame.loc[frame["split"].eq("adaptation")]
    if percent == 20:
        return candidates.copy()
    target_by_student = candidates.groupby("student_id")["target"].max()
    fraction = percent / 20
    selected, _ = train_test_split(
        target_by_student.index.to_numpy(),
        train_size=fraction,
        random_state=RANDOM_STATE + percent,
        stratify=target_by_student.to_numpy(),
    )
    return candidates.loc[
        candidates["student_id"].isin(selected)
    ].copy()


def institution_weights(frame):
    counts = frame["institution"].value_counts()
    return frame["institution"].map(
        len(frame) / (len(counts) * counts)
    ).to_numpy(dtype=float)


def training_weights(frame, balance_institutions):
    if balance_institutions:
        weights = institution_weights(frame)
    else:
        weights = np.ones(len(frame), dtype=float)

    target = frame["target"].to_numpy(dtype=int)
    positive_weight = weights[target == 1].sum()
    negative_weight = weights[target == 0].sum()
    if positive_weight > 0 and negative_weight > 0:
        weights[target == 1] *= 0.5 / positive_weight
        weights[target == 0] *= 0.5 / negative_weight
        weights *= len(weights) / weights.sum()
    return weights


def make_model(model_name):
    if model_name == "Logistic Regression":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
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
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=240,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=4,
                    reg_lambda=2.0,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def fit_model(train, model_name, balance_institutions):
    model = make_model(model_name)
    weights = training_weights(train, balance_institutions)
    model.fit(
        train[ENROLMENT_FEATURES],
        train["target"],
        model__sample_weight=weights,
    )
    return model


def evaluate_model(model, test, model_name, experiment):
    scores = model.predict_proba(test[ENROLMENT_FEATURES])[:, 1]
    metrics = {
        "feature_set": "enrolment",
        "model": model_name,
        "experiment": experiment,
        "train_target_definition": "eventual attrition",
        "test_institution": test["institution"].iloc[0],
        "test_rows": len(test),
        "test_students": test["student_id"].nunique(),
        "prevalence": test["target"].mean(),
        "roc_auc": roc_auc_score(test["target"], scores),
        "average_precision": average_precision_score(
            test["target"],
            scores,
        ),
    }
    predictions = test[
        ["institution", "record_id", "student_id", "target"]
    ].copy()
    predictions["feature_set"] = "enrolment"
    predictions["model"] = model_name
    predictions["experiment"] = experiment
    predictions["score"] = scores
    return metrics, predictions


def training_scenarios(uci, oulad):
    uci_source = uci.loc[uci["split"].isin(["train", "adaptation"])]
    oulad_source = oulad.loc[
        oulad["split"].isin(["train", "adaptation"])
    ]
    uci_test = uci.loc[uci["split"].eq("test")]
    oulad_test = oulad.loc[oulad["split"].eq("test")]
    pooled = pd.concat([uci_source, oulad_source], ignore_index=True)

    scenarios = [
        {
            "name": "uci_source",
            "train": uci_source,
            "balance_institutions": False,
            "tests": {
                "within_uci": uci_test,
                "zero_uci_to_oulad": oulad_test,
            },
        },
        {
            "name": "oulad_source",
            "train": oulad_source,
            "balance_institutions": False,
            "tests": {
                "within_oulad": oulad_test,
                "zero_oulad_to_uci": uci_test,
            },
        },
        {
            "name": "pooled",
            "train": pooled,
            "balance_institutions": True,
            "tests": {
                "pooled_to_uci": uci_test,
                "pooled_to_oulad": oulad_test,
            },
        },
    ]

    for percent in (5, 10, 20):
        oulad_local = select_adaptation_rows(oulad, percent)
        uci_local = select_adaptation_rows(uci, percent)
        scenarios.extend(
            [
                {
                    "name": f"adapt_uci_to_oulad_{percent:02d}",
                    "train": pd.concat(
                        [uci_source, oulad_local],
                        ignore_index=True,
                    ),
                    # Preserve the actual labeled-row contribution so a 5%
                    # local sample is not silently given half the total weight.
                    "balance_institutions": False,
                    "tests": {
                        f"adapt_uci_to_oulad_{percent:02d}": oulad_test
                    },
                },
                {
                    "name": f"adapt_oulad_to_uci_{percent:02d}",
                    "train": pd.concat(
                        [oulad_source, uci_local],
                        ignore_index=True,
                    ),
                    "balance_institutions": False,
                    "tests": {
                        f"adapt_oulad_to_uci_{percent:02d}": uci_test
                    },
                },
            ]
        )
    return scenarios


def run_experiments(uci, oulad):
    metric_rows = []
    prediction_frames = []
    final_model = None
    final_training = None

    for model_name in ("Logistic Regression", "XGBoost"):
        for scenario in training_scenarios(uci, oulad):
            model = fit_model(
                scenario["train"],
                model_name,
                scenario["balance_institutions"],
            )
            for experiment, test in scenario["tests"].items():
                metrics, predictions = evaluate_model(
                    model,
                    test,
                    model_name,
                    experiment,
                )
                metrics["train_rows"] = len(scenario["train"])
                metrics["training_scenario"] = scenario["name"]
                metrics["institution_balanced"] = scenario[
                    "balance_institutions"
                ]
                metric_rows.append(metrics)
                prediction_frames.append(predictions)

            if model_name == "XGBoost" and scenario["name"] == "pooled":
                final_model = model
                final_training = scenario["train"]

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, final_model, final_training


def metric_value(name, target, scores):
    if name == "roc_auc":
        return roc_auc_score(target, scores)
    return average_precision_score(target, scores)


def align_predictions(predictions, baseline, candidate):
    keep = predictions.loc[
        (predictions["feature_set"] == "enrolment")
        & (predictions["model"] == "XGBoost")
        & predictions["experiment"].isin([baseline, candidate])
    ]
    left = keep.loc[
        keep["experiment"].eq(baseline),
        ["record_id", "student_id", "target", "score"],
    ].rename(columns={"score": "baseline_score"})
    right = keep.loc[
        keep["experiment"].eq(candidate),
        ["record_id", "score"],
    ].rename(columns={"score": "candidate_score"})
    paired = left.merge(right, on="record_id", how="inner")
    if paired.empty:
        raise ValueError(
            f"No paired predictions for {baseline} and {candidate}."
        )
    return paired


def paired_test(predictions, baseline, candidate, label):
    paired = align_predictions(predictions, baseline, candidate)
    groups = paired["student_id"].unique()
    group_rows = {
        group: paired.index[paired["student_id"].eq(group)].to_numpy()
        for group in groups
    }
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for metric in ("roc_auc", "average_precision"):
        observed = metric_value(
            metric,
            paired["target"],
            paired["candidate_score"],
        ) - metric_value(
            metric,
            paired["target"],
            paired["baseline_score"],
        )

        bootstrap_values = []
        for _ in range(BOOTSTRAP_SAMPLES):
            sampled_groups = rng.choice(
                groups,
                size=len(groups),
                replace=True,
            )
            sampled_rows = np.concatenate(
                [group_rows[group] for group in sampled_groups]
            )
            sample = paired.loc[sampled_rows]
            if sample["target"].nunique() < 2:
                continue
            bootstrap_values.append(
                metric_value(
                    metric,
                    sample["target"],
                    sample["candidate_score"],
                )
                - metric_value(
                    metric,
                    sample["target"],
                    sample["baseline_score"],
                )
            )

        permutation_values = []
        for _ in range(PERMUTATION_SAMPLES):
            swap_groups = set(groups[rng.random(len(groups)) < 0.5])
            swap = paired["student_id"].isin(swap_groups).to_numpy()
            baseline_scores = np.where(
                swap,
                paired["candidate_score"],
                paired["baseline_score"],
            )
            candidate_scores = np.where(
                swap,
                paired["baseline_score"],
                paired["candidate_score"],
            )
            permutation_values.append(
                metric_value(
                    metric,
                    paired["target"],
                    candidate_scores,
                )
                - metric_value(
                    metric,
                    paired["target"],
                    baseline_scores,
                )
            )

        bootstrap_values = np.asarray(bootstrap_values)
        permutation_values = np.asarray(permutation_values)
        p_value = (
            1 + np.sum(np.abs(permutation_values) >= abs(observed))
        ) / (PERMUTATION_SAMPLES + 1)
        rows.append(
            {
                "comparison": label,
                "baseline": baseline,
                "candidate": candidate,
                "metric": metric,
                "difference": observed,
                "ci_low": np.quantile(bootstrap_values, 0.025),
                "ci_high": np.quantile(bootstrap_values, 0.975),
                "permutation_p": p_value,
                "test_rows": len(paired),
                "test_students": len(groups),
            }
        )
    return rows


def run_paired_tests(predictions):
    comparisons = [
        (
            "zero_uci_to_oulad",
            "adapt_uci_to_oulad_20",
            "OULAD: zero-shot vs 20% local adaptation",
        ),
        (
            "zero_oulad_to_uci",
            "adapt_oulad_to_uci_20",
            "UCI: zero-shot vs 20% local adaptation",
        ),
        (
            "within_oulad",
            "pooled_to_oulad",
            "OULAD: local-only vs pooled model",
        ),
        (
            "within_uci",
            "pooled_to_uci",
            "UCI: local-only vs pooled model",
        ),
    ]
    rows = []
    for baseline, candidate, label in comparisons:
        rows.extend(paired_test(predictions, baseline, candidate, label))
    return pd.DataFrame(rows)


def calculate_shared_shap(model, uci, oulad):
    test_sets = {
        "UCI_dataset": uci.loc[uci["split"].eq("test")],
        "OULAD_dataset": oulad.loc[oulad["split"].eq("test")],
    }
    imputer = model.named_steps["imputer"]
    xgboost_model = model.named_steps["model"]
    explainer = shap.TreeExplainer(xgboost_model)
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)

    for institution, frame in test_sets.items():
        if len(frame) > 2000:
            selected = rng.choice(frame.index, size=2000, replace=False)
            frame = frame.loc[selected]
        values = imputer.transform(frame[ENROLMENT_FEATURES])
        shap_values = np.asarray(explainer.shap_values(values))
        mean_absolute = np.abs(shap_values).mean(axis=0)
        for feature, importance in zip(
            ENROLMENT_FEATURES,
            mean_absolute,
        ):
            rows.append(
                {
                    "institution": institution,
                    "shared_feature": feature,
                    "mean_absolute_shap": importance,
                }
            )
    return pd.DataFrame(rows)


def plot_transfer_results(metrics):
    selected = metrics.loc[
        (metrics["feature_set"] == "enrolment")
        & (metrics["model"] == "XGBoost")
    ].set_index("experiment")
    labels = ["Local only", "Zero-shot", "+20% local", "Pooled"]
    experiments = {
        "UCI dataset test": [
            "within_uci",
            "zero_oulad_to_uci",
            "adapt_oulad_to_uci_20",
            "pooled_to_uci",
        ],
        "OULAD dataset test": [
            "within_oulad",
            "zero_uci_to_oulad",
            "adapt_uci_to_oulad_20",
            "pooled_to_oulad",
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    colors = ["#0F4C5C", "#C65D3B", "#E6A33D", "#5B8E7D"]
    for axis, (title, names) in zip(axes, experiments.items()):
        values = [selected.loc[name, "roc_auc"] for name in names]
        bars = axis.bar(labels, values, color=colors)
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_ylim(0.40, 0.85)
        axis.set_ylabel("ROC-AUC")
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle(
        "Cross-dataset portability of the shared enrolment model"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "unified_cross_school_transfer.png", dpi=220)
    plt.close(fig)


def plot_adaptation_curve(metrics):
    selected = metrics.loc[
        (metrics["feature_set"] == "enrolment")
        & (metrics["model"] == "XGBoost")
    ].set_index("experiment")
    percentages = [0, 5, 10, 20]
    oulad_values = [selected.loc["zero_uci_to_oulad", "roc_auc"]] + [
        selected.loc[f"adapt_uci_to_oulad_{value:02d}", "roc_auc"]
        for value in percentages[1:]
    ]
    uci_values = [selected.loc["zero_oulad_to_uci", "roc_auc"]] + [
        selected.loc[f"adapt_oulad_to_uci_{value:02d}", "roc_auc"]
        for value in percentages[1:]
    ]

    fig, axis = plt.subplots(figsize=(7.5, 4.6))
    axis.plot(percentages, uci_values, marker="o", label="Target: UCI")
    axis.plot(
        percentages,
        oulad_values,
        marker="o",
        label="Target: OULAD",
    )
    axis.set_xlabel("Labeled target-dataset records used (%)")
    axis.set_ylabel("ROC-AUC on untouched target test set")
    axis.set_xticks(percentages)
    axis.set_ylim(0.4, 0.85)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    axis.set_title("Local labels are an adaptation experiment, not zero-shot")
    fig.tight_layout()
    fig.savefig(FIGURES / "unified_adaptation_curve.png", dpi=220)
    plt.close(fig)


def plot_shared_shap(shap_results):
    pivot = shap_results.pivot(
        index="shared_feature",
        columns="institution",
        values="mean_absolute_shap",
    ).loc[ENROLMENT_FEATURES]
    pivot = pivot.sort_values("OULAD_dataset")
    axis = pivot.plot.barh(
        figsize=(8.2, 5.2),
        color=["#0F4C5C", "#E6A33D"],
    )
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Shared enrolment feature")
    axis.set_title("One pooled model evaluated in two data contexts")
    axis.legend(title="Test dataset")
    plt.tight_layout()
    plt.savefig(FIGURES / "unified_shared_feature_shap.png", dpi=220)
    plt.close()


def save_model_artifact(model, training_frame):
    MODELS.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": model,
        "schema_version": SCHEMA_VERSION,
        "features": ENROLMENT_FEATURES,
        "core_features": CORE_FEATURES,
        "optional_features": OPTIONAL_FEATURES,
        "observation_point": "enrolment or registration",
        "target_definition": "eventual attrition",
        "score_semantics": (
            "uncalibrated ranking score; do not present as an individual "
            "withdrawal probability"
        ),
        "training_datasets": sorted(
            training_frame["institution"].unique().tolist()
        ),
        "training_rows": len(training_frame),
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
        "optional_missing_values": "median imputation from training data",
        "recommended_output": "within-course or within-cohort percentile",
        "default_review_share": 0.15,
        "external_validation_required": True,
    }
    CONTRACT_PATH.write_text(
        json.dumps(contract, indent=2),
        encoding="utf-8",
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("Step 1: Build enrolment-time dataset adapters")
    uci = make_uci_adapter()
    oulad = make_oulad_adapter()
    validate_adapter(uci)
    validate_adapter(oulad)
    uci = assign_splits(uci)
    oulad = assign_splits(oulad)

    print("Step 2: Save the shared feature contract")
    pd.DataFrame(SCHEMA_ROWS).to_csv(
        RESULTS / "unified_shared_feature_schema.csv",
        index=False,
    )

    print("Step 3: Run local, zero-shot, pooled, and adaptation tests")
    metrics, predictions, final_model, final_training = run_experiments(
        uci,
        oulad,
    )
    metrics.to_csv(RESULTS / "unified_model_metrics.csv", index=False)
    predictions.to_csv(
        RESULTS / "unified_model_predictions.csv",
        index=False,
    )

    print("Step 4: Run paired student-cluster tests")
    significance = run_paired_tests(predictions)
    significance.to_csv(
        RESULTS / "unified_model_significance.csv",
        index=False,
    )

    print("Step 5: Explain and persist the single pooled model")
    shared_shap = calculate_shared_shap(final_model, uci, oulad)
    shared_shap.to_csv(
        RESULTS / "unified_shared_feature_shap.csv",
        index=False,
    )
    save_model_artifact(final_model, final_training)

    print("Step 6: Save figures")
    plot_transfer_results(metrics)
    plot_adaptation_curve(metrics)
    plot_shared_shap(shared_shap)

    key_rows = metrics.loc[
        (metrics["model"] == "XGBoost")
        & metrics["experiment"].isin(
            [
                "within_uci",
                "within_oulad",
                "zero_uci_to_oulad",
                "zero_oulad_to_uci",
                "pooled_to_uci",
                "pooled_to_oulad",
            ]
        )
    ]
    print("\nMain portability results")
    print(
        key_rows[
            [
                "experiment",
                "test_rows",
                "prevalence",
                "roc_auc",
                "average_precision",
            ]
        ].round(4).to_string(index=False)
    )
    print("\nSaved model:", MODEL_PATH)


if __name__ == "__main__":
    main()
