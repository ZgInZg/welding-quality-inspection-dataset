#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from common import generate, load_text, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the diagnosis or optimization LLM judge")
    parser.add_argument("--task", choices=["diagnosis", "optimization"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def stringify(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def fill(template: str, record: dict[str, object], keys: list[str]) -> str:
    for key in keys:
        if key not in record:
            raise ValueError(f"Missing field {key!r} in sample {record.get('sample_id', '<unknown>')}")
        template = template.replace("{" + key + "}", stringify(record[key]))
    return template


def parse_json_object(text: str) -> dict[str, object]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge response does not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Judge response must decode to a JSON object")
    return value


def main() -> None:
    args = parse_args()
    if args.task == "diagnosis":
        template = load_text(ROOT / "prompts/A4_diagnosis_judge.txt")
        keys = ["ground_truth", "prediction"]
    else:
        template = load_text(ROOT / "prompts/A5_optimization_judge.txt")
        keys = [
            "diagnosis",
            "current_parameters",
            "reference_parameters",
            "physics_rules",
            "optimization_proposal",
        ]
    outputs = []
    for record in read_jsonl(args.input):
        prompt = fill(template, record, keys)
        result = generate(prompt, env_prefix="JUDGE")
        outputs.append({**record, "judge_output": result, "judge_scores": parse_json_object(result)})
    write_jsonl(args.output, outputs)
    print(f"Wrote {len(outputs)} judge outputs to {args.output}")


if __name__ == "__main__":
    main()
