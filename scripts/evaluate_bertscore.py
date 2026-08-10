#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bert_score import score

from common import substantive_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sample-level and grouped BERTScore")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--group-by", nargs="*", default=["dataset", "model", "strategy"])
    parser.add_argument("--model-type", default="roberta-large")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {args.reference_column, args.prediction_column}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    references = [substantive_text(value) for value in frame[args.reference_column]]
    predictions = [substantive_text(value) for value in frame[args.prediction_column]]
    if any(not value for value in references + predictions):
        raise SystemExit("Reference and prediction texts must be non-empty after JSON cleaning")
    precision, recall, f1 = score(
        predictions,
        references,
        model_type=args.model_type,
        lang="en",
        device=args.device,
        verbose=True,
        rescale_with_baseline=False,
    )
    sample_frame = frame.copy()
    sample_frame["bertscore_precision"] = precision.cpu().numpy()
    sample_frame["bertscore_recall"] = recall.cpu().numpy()
    sample_frame["bertscore_f1"] = f1.cpu().numpy()
    group_columns = [column for column in args.group_by if column in sample_frame.columns]
    metric_columns = ["bertscore_precision", "bertscore_recall", "bertscore_f1"]
    if group_columns:
        summary = sample_frame.groupby(group_columns, dropna=False)[metric_columns].mean().reset_index()
        summary["n"] = sample_frame.groupby(group_columns, dropna=False).size().to_numpy()
    else:
        summary = pd.DataFrame([{**sample_frame[metric_columns].mean().to_dict(), "n": len(sample_frame)}])
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    sample_path = Path(f"{args.output_prefix}_sample.csv")
    summary_path = Path(f"{args.output_prefix}_summary.csv")
    sample_frame.to_csv(sample_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
