from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd

from cross_school_dynamic_model import (
    CORE_FEATURES,
    MODEL_FEATURES,
    MODEL_PATH,
    OPTIONAL_FEATURES,
    prepare_scoring_frame,
)


def score_frame(
    frame,
    artifact,
    group_column=None,
    review_share=0.15,
):
    prepared = prepare_scoring_frame(frame)
    expected = artifact["features"]
    if expected != MODEL_FEATURES:
        raise ValueError(
            "The saved model feature order does not match the scoring code."
        )
    if artifact["core_features"] != CORE_FEATURES:
        raise ValueError(
            "The saved model core features do not match the scoring code."
        )

    scores = artifact["pipeline"].predict_proba(
        prepared[expected]
    )[:, 1]
    output_columns = [
        column
        for column in (
            "institution",
            "record_id",
            "student_id",
            "course_id",
            "snapshot_day",
        )
        if column in prepared
    ]
    output = prepared[output_columns].copy()
    output["risk_score"] = scores

    if group_column:
        group_columns = [
            column.strip()
            for column in group_column.split(",")
            if column.strip()
        ]
        missing = [
            column for column in group_columns if column not in prepared
        ]
        if missing:
            raise ValueError(
                "Grouping columns were not found: " + ", ".join(missing)
            )
        output["risk_percentile"] = output.groupby(
            group_columns,
            sort=False,
        )["risk_score"].rank(method="average", pct=True)
    else:
        output["risk_percentile"] = output["risk_score"].rank(
            method="average",
            pct=True,
        )

    output["review_flag"] = output["risk_percentile"].ge(
        1 - review_share
    )
    available_optional = prepared[OPTIONAL_FEATURES].notna().sum(axis=1)
    output["optional_feature_coverage"] = (
        available_optional / len(OPTIONAL_FEATURES)
    )
    return output.sort_values("risk_score", ascending=False)


def feature_coverage(frame):
    prepared = prepare_scoring_frame(frame)
    return pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "required": [
                feature in CORE_FEATURES for feature in MODEL_FEATURES
            ],
            "available_share": [
                prepared[feature].notna().mean()
                for feature in MODEL_FEATURES
            ],
        }
    )


def parse_args():
    parser = ArgumentParser(
        description=(
            "Score fixed-schema enrolment or in-course snapshots with the "
            "frozen portable dynamic model. The score is a ranking value, "
            "not a calibrated probability."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the saved dynamic-model joblib artifact.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help=(
            "Optional comma-separated columns for local percentiles, such "
            "as course_id,snapshot_day."
        ),
    )
    parser.add_argument(
        "--review-share",
        type=float,
        default=0.15,
        help="Share of highest-ranked records flagged for review.",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=None,
        help="Optional CSV path for feature-availability diagnostics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.review_share < 1:
        raise ValueError("--review-share must be between 0 and 1.")
    artifact = joblib.load(args.model)
    frame = pd.read_csv(args.input_csv)
    scored = score_frame(
        frame,
        artifact,
        group_column=args.group_column,
        review_share=args.review_share,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_csv, index=False)

    coverage = feature_coverage(frame)
    if args.coverage_output:
        args.coverage_output.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(args.coverage_output, index=False)

    unavailable = coverage.loc[
        (~coverage["required"])
        & coverage["available_share"].eq(0),
        "feature",
    ].tolist()
    print(f"Scored {len(scored)} records.")
    print(f"Saved {args.output_csv}")
    if unavailable:
        print("Optional fields unavailable for every row:")
        print(", ".join(unavailable))
    print(
        "Important: risk_score is an uncalibrated ranking score, "
        "not an individual withdrawal probability."
    )


if __name__ == "__main__":
    main()
