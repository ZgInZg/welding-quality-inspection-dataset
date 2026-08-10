#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sample-level 1-5 LLM judge scores")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--group-by", nargs="*", default=["task", "dataset", "model", "strategy", "dimension"])
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".jsonl":
        return pd.read_csv(path)
    rows = []
    dimension_map = {
        "accuracy_score": "accuracy",
        "fidelity_score": "fidelity",
        "terminology_score": "terminology",
        "Dimension_1": "physical_compliance",
        "Dimension_2": "reasoning_specificity",
        "Dimension_3": "adaptive_calibration",
    }
    for record in read_jsonl(path):
        scores = record.get("judge_scores")
        if not isinstance(scores, dict):
            raise ValueError(f"Sample {record.get('sample_id', '<unknown>')} has no judge_scores object")
        for source_key, dimension in dimension_map.items():
            if source_key in scores:
                rows.append(
                    {
                        **{key: value for key, value in record.items() if key not in {"judge_output", "judge_scores"}},
                        "dimension": dimension,
                        "score": scores[source_key],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frame = load_frame(args.input)
    required = {"sample_id", "dimension", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    scores = pd.to_numeric(frame["score"], errors="raise")
    if not scores.between(1, 5).all():
        raise SystemExit("All judge scores must be within the 1-5 Likert range")
    if not (scores % 1 == 0).all():
        raise SystemExit("Sample-level judge scores must be ordinal integers")
    frame = frame.copy()
    frame["score"] = scores.astype(int)
    group_columns = [column for column in args.group_by if column in frame.columns]
    mean_frame = frame.groupby(group_columns, dropna=False)["score"].agg(["count", "mean", "std"]).reset_index()
    distribution = (
        frame.groupby(group_columns + ["score"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = distribution.groupby(group_columns, dropna=False)["count"].transform("sum")
    distribution["percent"] = 100.0 * distribution["count"] / totals
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    mean_frame.to_csv(Path(f"{args.output_prefix}_means.csv"), index=False)
    distribution.to_csv(Path(f"{args.output_prefix}_likert_distribution.csv"), index=False)
    print(mean_frame.to_string(index=False))


if __name__ == "__main__":
    main()
