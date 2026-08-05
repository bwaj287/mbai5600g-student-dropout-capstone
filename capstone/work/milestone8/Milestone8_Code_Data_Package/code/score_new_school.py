from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd

from cross_school_unified_model import (
    ENROLMENT_FEATURES,
    MODEL_PATH,
    prepare_scoring_frame,
)


def score_frame(
    frame,
    artifact,
    group_column=None,
    review_share=0.15,
):
    prepared = prepare_scoring_frame(frame)
    pipeline = artifact["pipeline"]
    expected = artifact["features"]
    if expected != ENROLMENT_FEATURES:
        raise ValueError(
            "The model feature order does not match the scoring code."
        )

    scores = pipeline.predict_proba(prepared[expected])[:, 1]
    output_columns = [
        column
        for column in (
            "institution",
            "record_id",
            "student_id",
            "course_id",
        )
        if column in prepared
    ]
    output = prepared[output_columns].copy()
    output["risk_score"] = scores

    if group_column:
        if group_column not in prepared:
            raise ValueError(
                f"Grouping column was not found: {group_column}"
            )
        output["risk_percentile"] = output.groupby(
            prepared[group_column],
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
    return output.sort_values("risk_score", ascending=False)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Score enrolment-time records with the frozen pooled model. "
            "The score is a ranking value, not a calibrated probability."
        )
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the saved joblib artifact.",
    )
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional column for within-course/cohort percentiles.",
    )
    parser.add_argument(
        "--review-share",
        type=float,
        default=0.15,
        help="Share of highest-ranked records flagged for review.",
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
    print(f"Scored {len(scored)} records.")
    print(f"Saved {args.output_csv}")
    print(
        "Important: risk_score is an uncalibrated ranking score, "
        "not an individual withdrawal probability."
    )


if __name__ == "__main__":
    main()
