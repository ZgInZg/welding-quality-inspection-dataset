#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import generate, load_text, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physics-guided parameter optimization")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/A3_parameter_optimization.txt")
    return parser.parse_args()


def resolve(path: str, input_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (input_path.parent / candidate).resolve()


def stringify(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    template = load_text(args.prompt)
    output_records = []
    for record in read_jsonl(args.input):
        required = {
            "sample_id",
            "query_image",
            "diagnosis",
            "current_parameters",
            "reference_parameters",
            "physics_rules",
        }
        missing = required - set(record)
        if missing:
            raise SystemExit(f"Record {record.get('sample_id', '<unknown>')} is missing {sorted(missing)}")
        prompt = template
        replacements = {
            "query_image": "the attached query image",
            "diagnosis": stringify(record["diagnosis"]),
            "current_parameters": stringify(record["current_parameters"]),
            "reference_parameters": stringify(record["reference_parameters"]),
            "physics_rules": stringify(record["physics_rules"]),
        }
        for key, value in replacements.items():
            prompt = prompt.replace("{" + key + "}", value)
        prediction = generate(prompt, images=[resolve(record["query_image"], args.input)])
        output_records.append({**record, "optimization_proposal": prediction})
    write_jsonl(args.output, output_records)
    print(f"Wrote {len(output_records)} proposals to {args.output}")


if __name__ == "__main__":
    main()
