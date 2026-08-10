#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from common import mae, mean, pearson_r, weighted_kappa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute inter-expert and LLM-human agreement")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--group-by", nargs="*", default=["task", "dataset"])
    parser.add_argument("--expert-columns", nargs="+", default=["expert_1", "expert_2", "expert_3"])
    parser.add_argument("--llm-column", default="llm_score")
    return parser.parse_args()


def agreement(left: list[float], right: list[float]) -> dict[str, float]:
    return {
        "pearson_r": pearson_r(left, right),
        "weighted_kappa": weighted_kappa(left, right),
        "mae": mae(left, right),
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {args.llm_column, *args.expert_columns}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].between(1, 5).all():
            raise SystemExit(f"{column} contains values outside the 1-5 range")
    group_columns = [column for column in args.group_by if column in frame.columns]
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), frame)]
    inter_rows: list[dict[str, object]] = []
    llm_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for key, group in groups:
        values = key if isinstance(key, tuple) else (key,)
        identity = dict(zip(group_columns, values))
        pair_metrics = []
        for first, second in combinations(args.expert_columns, 2):
            metrics = agreement(group[first].tolist(), group[second].tolist())
            pair_rows.append({**identity, "expert_pair": f"{first} vs {second}", "n": len(group), **metrics})
            pair_metrics.append(metrics)
        inter_rows.append(
            {
                **identity,
                "n": len(group),
                "mean_pairwise_r": mean(row["pearson_r"] for row in pair_metrics),
                "mean_pairwise_weighted_kappa": mean(row["weighted_kappa"] for row in pair_metrics),
                "mean_pairwise_mae": mean(row["mae"] for row in pair_metrics),
            }
        )
        human_mean = group[args.expert_columns].mean(axis=1).tolist()
        llm_metrics = agreement(group[args.llm_column].tolist(), human_mean)
        llm_rows.append({**identity, "n": len(group), **llm_metrics})

    inter_frame = pd.DataFrame(inter_rows)
    llm_frame = pd.DataFrame(llm_rows)
    if len(inter_frame) > 1:
        inter_frame = pd.concat(
            [
                inter_frame,
                pd.DataFrame(
                    [
                        {
                            **{column: "Overall Mean" if index == 0 else "" for index, column in enumerate(group_columns)},
                            "n": inter_frame["n"].sum(),
                            "mean_pairwise_r": inter_frame["mean_pairwise_r"].mean(),
                            "mean_pairwise_weighted_kappa": inter_frame["mean_pairwise_weighted_kappa"].mean(),
                            "mean_pairwise_mae": inter_frame["mean_pairwise_mae"].mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        llm_frame = pd.concat(
            [
                llm_frame,
                pd.DataFrame(
                    [
                        {
                            **{column: "Overall Mean" if index == 0 else "" for index, column in enumerate(group_columns)},
                            "n": llm_frame["n"].sum(),
                            "pearson_r": llm_frame["pearson_r"].mean(),
                            "weighted_kappa": llm_frame["weighted_kappa"].mean(),
                            "mae": llm_frame["mae"].mean(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pair_rows).to_csv(Path(f"{args.output_prefix}_expert_pairs.csv"), index=False)
    inter_frame.to_csv(Path(f"{args.output_prefix}_inter_expert.csv"), index=False)
    llm_frame.to_csv(Path(f"{args.output_prefix}_llm_vs_human.csv"), index=False)
    print("Inter-expert agreement")
    print(inter_frame.to_string(index=False))
    print("\nLLM-human agreement")
    print(llm_frame.to_string(index=False))


if __name__ == "__main__":
    main()
