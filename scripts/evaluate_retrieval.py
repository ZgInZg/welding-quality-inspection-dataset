#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import binary_metrics


WEIGHT_SCHEMES = {
    1: [1.0],
    3: [5.0, 3.0, 2.0],
    5: [8.0, 5.0, 4.0, 2.0, 1.0],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Top-k retrieval voting")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, choices=[1, 3, 5], default=5)
    parser.add_argument("--voting", choices=["equal", "weighted"], default="weighted")
    parser.add_argument("--threshold", type=float, default=0.68)
    return parser.parse_args()


def label(value: object) -> int:
    text = str(value).strip().upper()
    if text not in {"OK", "NG"}:
        raise ValueError(f"Expected OK or NG, received {value!r}")
    return int(text == "OK")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"query_id", "query_label", "reference_label", "rank", "cosine_similarity"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    frame = frame[frame["rank"] <= args.top_k].copy()
    frame["weight"] = 1.0
    if args.voting == "weighted":
        rank_weights = dict(enumerate(WEIGHT_SCHEMES[args.top_k], start=1))
        frame["weight"] = frame["rank"].map(rank_weights)
    prediction_rows = []
    for query_id, group in frame.groupby("query_id", sort=False):
        expected_ranks = set(range(1, args.top_k + 1))
        if set(group["rank"].astype(int)) != expected_ranks:
            raise SystemExit(f"Query {query_id} does not contain ranks 1..{args.top_k}")
        ng_weight = group.loc[group["reference_label"].str.upper() == "NG", "weight"].sum()
        ok_weight = group.loc[group["reference_label"].str.upper() == "OK", "weight"].sum()
        max_similarity = float(group["cosine_similarity"].max())
        rejected = max_similarity < args.threshold
        prediction_rows.append(
            {
                "query_id": query_id,
                "y_true": label(group["query_label"].iloc[0]),
                "y_pred": 0 if rejected else int(ok_weight >= ng_weight),
                "rejected": rejected,
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    metrics = pd.DataFrame(
        [
            {
                "top_k": args.top_k,
                "voting": args.voting,
                "threshold": args.threshold,
                "rejected_count": int(predictions["rejected"].sum()),
                **binary_metrics(predictions["y_true"], predictions["y_pred"]),
            }
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
