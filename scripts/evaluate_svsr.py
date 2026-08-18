#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute safety-veto success rate by defect multiplicity")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def as_flag(value: object) -> int:
    text = str(value).strip().lower()
    mapping = {"0": 0, "false": 0, "safe": 0, "1": 1, "true": 1, "unsafe": 1}
    if text not in mapping:
        raise ValueError(f"Unsupported safety flag: {value!r}")
    return mapping[text]


def summarize(group: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    n_unique_samples = int(group["sample_id"].nunique())
    n_evaluations = len(group)
    before_count = int(group["_before"].sum())
    residual_count = int(group["_after"].sum())
    veto_success_count = int(((group["_before"] == 1) & (group["_after"] == 0)).sum())
    new_unsafe_count = int(((group["_before"] == 0) & (group["_after"] == 1)).sum())
    veto_trigger_success = 100.0 * veto_success_count / before_count if before_count else 0.0
    net_reduction = 100.0 * (before_count - residual_count) / before_count if before_count else 0.0
    return {
        "defect_group": label,
        "n": n_unique_samples,
        "n_evaluations": n_evaluations,
        "unsafe_before_count": before_count,
        "unsafe_before_percent": 100.0 * before_count / n_evaluations if n_evaluations else 0.0,
        "residual_unsafe_count": residual_count,
        "residual_unsafe_percent": 100.0 * residual_count / n_evaluations if n_evaluations else 0.0,
        "veto_success_count": veto_success_count,
        "new_unsafe_count": new_unsafe_count,
        "veto_trigger_success_percent": veto_trigger_success,
        "svsr_percent": net_reduction,
    }


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    required = {"sample_id", "defect_count", "unsafe_before", "unsafe_after"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    frame["defect_count"] = pd.to_numeric(frame["defect_count"], errors="raise").astype(int)
    if (frame["defect_count"] < 1).any():
        raise SystemExit("SVSR analysis requires defect-bearing samples with defect_count >= 1")
    frame["_before"] = frame["unsafe_before"].map(as_flag)
    frame["_after"] = frame["unsafe_after"].map(as_flag)
    rows = [
        summarize(frame[frame["defect_count"] == 1], "Single-defect"),
        summarize(frame[frame["defect_count"] > 1], "Multi-defect"),
        summarize(frame, "Defect-bearing overall"),
    ]
    summary = pd.DataFrame(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(f"{args.output_prefix}_summary.csv"), index=False)
    if "failure_type" in frame.columns:
        residual = frame[frame["_after"] == 1].copy()
        failures = residual.groupby("failure_type", dropna=False).size().rename("count").reset_index()
        failures["percent"] = 100.0 * failures["count"] / len(residual) if len(residual) else 0.0
        failures.to_csv(Path(f"{args.output_prefix}_failure_types.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
