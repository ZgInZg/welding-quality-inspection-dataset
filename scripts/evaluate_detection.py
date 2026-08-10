#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute binary defect-detection metrics")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-by", nargs="*", default=["dataset", "model", "strategy"])
    return parser.parse_args()


def normalize_label(value: object) -> int:
    text = str(value).strip().lower()
    mapping = {"0": 0, "ok": 0, "normal": 0, "negative": 0, "1": 1, "ng": 1, "defect": 1, "positive": 1}
    if text not in mapping:
        raise ValueError(f"Unsupported binary label: {value!r}")
    return mapping[text]


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"y_true", "y_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    frame["_true"] = frame["y_true"].map(normalize_label)
    frame["_pred"] = frame["y_pred"].map(normalize_label)
    group_columns = [column for column in args.group_by if column in frame.columns]
    groups = frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [((), frame)]
    rows = []
    for key, group in groups:
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, values))
        row.update(binary_metrics(group["_true"].tolist(), group["_pred"].tolist()))
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
