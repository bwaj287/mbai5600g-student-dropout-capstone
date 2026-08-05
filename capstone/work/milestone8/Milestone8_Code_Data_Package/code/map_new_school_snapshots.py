from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import pandas as pd

from cross_school_dynamic_model import (
    MODEL_FEATURES,
    OPTIONAL_FEATURES,
    prepare_scoring_frame,
)


METADATA_COLUMNS = [
    "institution",
    "record_id",
    "student_id",
    "course_id",
    "snapshot_day",
]


def source_series(raw, columns, name, required=False):
    source = columns.get(name)
    if source is None:
        if required:
            raise ValueError(f"Mapping is required for: {name}")
        return pd.Series(np.nan, index=raw.index, dtype=float)
    if source not in raw:
        raise ValueError(f"Mapped source column was not found: {source}")
    return raw[source]


def mapped_numeric(raw, config, feature, required=False):
    values = source_series(
        raw,
        config["columns"],
        feature,
        required=required,
    )
    value_map = config.get("value_maps", {}).get(feature)
    if value_map:
        values = values.astype("string").map(
            {str(key): value for key, value in value_map.items()}
        )
    return pd.to_numeric(values, errors="coerce")


def build_standard_frame(raw, config):
    columns = config.get("columns", {})
    scales = config.get("scales", {})
    output = pd.DataFrame(index=raw.index)

    institution = config.get("institution", "new_school")
    output["institution"] = institution
    if "institution" in columns:
        output["institution"] = source_series(
            raw,
            columns,
            "institution",
        ).astype(str)

    if "record_id" in columns:
        output["record_id"] = source_series(
            raw,
            columns,
            "record_id",
        ).astype(str)
    else:
        output["record_id"] = "ROW_" + raw.index.astype(str)

    if "student_id" in columns:
        output["student_id"] = source_series(
            raw,
            columns,
            "student_id",
        ).astype(str)
    else:
        output["student_id"] = output["record_id"]

    if "course_id" in columns:
        output["course_id"] = source_series(
            raw,
            columns,
            "course_id",
        ).astype(str)

    snapshot_day = mapped_numeric(
        raw,
        config,
        "snapshot_day",
        required=True,
    )
    if snapshot_day.isna().any() or snapshot_day.lt(0).any():
        raise ValueError("snapshot_day must be a non-negative number.")
    output["snapshot_day"] = snapshot_day

    if "age_scaled" in columns:
        output["age_scaled"] = mapped_numeric(
            raw,
            config,
            "age_scaled",
            required=True,
        )
    else:
        age = mapped_numeric(raw, config, "age", required=True)
        age_min = float(scales.get("age_min", 18))
        age_max = float(scales.get("age_max", 60))
        if age_max <= age_min:
            raise ValueError("age_max must be greater than age_min.")
        output["age_scaled"] = (
            (age - age_min) / (age_max - age_min)
        ).clip(0, 1)

    output["prior_education_level"] = mapped_numeric(
        raw,
        config,
        "prior_education_level",
        required=True,
    )

    study_load = mapped_numeric(
        raw,
        config,
        "study_load",
        required=True,
    )
    if study_load.isna().any() or study_load.lt(0).any():
        raise ValueError("study_load must be a non-negative number.")
    heavy_load = float(scales.get("heavy_study_load", 1.0))
    if heavy_load <= 0:
        raise ValueError("heavy_study_load must be positive.")
    output["study_load"] = (study_load / heavy_load).clip(0, 1)

    if "course_progress_ratio" in columns:
        output["course_progress_ratio"] = mapped_numeric(
            raw,
            config,
            "course_progress_ratio",
            required=True,
        )
    else:
        length_column = columns.get("course_length_days")
        if length_column:
            course_length = pd.to_numeric(
                raw[length_column],
                errors="coerce",
            )
        else:
            fixed_length = scales.get("course_length_days")
            if fixed_length is None:
                raise ValueError(
                    "Map course_progress_ratio or provide course_length_days."
                )
            course_length = pd.Series(
                float(fixed_length),
                index=raw.index,
            )
        if course_length.isna().any() or course_length.le(0).any():
            raise ValueError("course_length_days must be positive.")
        output["course_progress_ratio"] = (
            snapshot_day / course_length
        ).clip(upper=1)

    prior_score = mapped_numeric(
        raw,
        config,
        "prior_academic_score",
    )
    prior_score_max = float(scales.get("prior_academic_score_max", 1.0))
    if prior_score_max <= 0:
        raise ValueError("prior_academic_score_max must be positive.")
    output["prior_academic_score"] = (
        prior_score / prior_score_max
    ).clip(0, 1)

    attempts = mapped_numeric(raw, config, "previous_attempts")
    attempts_cap = float(scales.get("previous_attempts_cap", 1.0))
    if attempts_cap <= 0:
        raise ValueError("previous_attempts_cap must be positive.")
    output["previous_attempts"] = (
        attempts / attempts_cap
    ).clip(0, 1)

    direct_features = [
        "male",
        "declared_support_need",
        "financial_stability",
        "assessment_completion_rate",
        "assessment_average_score",
        "assessment_score_available",
        "late_submission_rate",
        "active_day_rate",
        "recent_activity_rate",
    ]
    for feature in direct_features:
        output[feature] = mapped_numeric(raw, config, feature)

    for feature in (
        "male",
        "declared_support_need",
        "assessment_score_available",
    ):
        observed = output[feature].dropna()
        if not observed.isin([0, 1]).all():
            raise ValueError(f"{feature} must contain only 0, 1, or blank.")

    inactivity = mapped_numeric(
        raw,
        config,
        "days_since_last_activity_scaled",
    )
    inactivity_cap = float(
        scales.get("days_since_last_activity_cap", 1.0)
    )
    if inactivity_cap <= 0:
        raise ValueError("days_since_last_activity_cap must be positive.")
    output["days_since_last_activity_scaled"] = (
        inactivity / inactivity_cap
    ).clip(upper=1)

    for feature in OPTIONAL_FEATURES:
        if feature not in output:
            output[feature] = np.nan
    output = output[
        [column for column in METADATA_COLUMNS if column in output]
        + MODEL_FEATURES
    ]
    return prepare_scoring_frame(output)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Map local snapshot columns to the fixed portable dynamic schema."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("mapping_json", type=Path)
    parser.add_argument("output_csv", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    raw = pd.read_csv(args.input_csv)
    config = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    mapped = build_standard_frame(raw, config)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(args.output_csv, index=False)
    print(f"Mapped {len(mapped)} snapshot rows.")
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
