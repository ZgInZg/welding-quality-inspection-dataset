#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import generate, load_text, read_jsonl, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference-guided defect diagnosis")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=ROOT / "prompts/A2_defect_diagnosis.txt")
    return parser.parse_args()


def resolve(path: str, input_path: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (input_path.parent / candidate).resolve()


def substitute(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def main() -> None:
    args = parse_args()
    template = load_text(args.prompt)
    output_records = []
    for record in read_jsonl(args.input):
        required = {"sample_id", "query_image", "standard_image", "defect_image", "reference_diagnosis"}
        missing = required - set(record)
        if missing:
            raise SystemExit(f"Record {record.get('sample_id', '<unknown>')} is missing {sorted(missing)}")
        diagnosis_value = record["reference_diagnosis"]
        if isinstance(diagnosis_value, str) and Path(diagnosis_value).suffix.lower() == ".json":
            diagnosis_value = json.loads(resolve(diagnosis_value, args.input).read_text(encoding="utf-8"))
        prompt = substitute(
            template,
            {
                "standard_image": "the attached standard-reference image",
                "defect_image": "the attached defect-reference image",
                "reference_diagnosis": json.dumps(diagnosis_value, ensure_ascii=False),
                "query_image": "the attached query image",
            },
        )
        images = [
            resolve(record["standard_image"], args.input),
            resolve(record["defect_image"], args.input),
            resolve(record["query_image"], args.input),
        ]
        prediction = generate(prompt, images=images)
        output_records.append({**record, "prediction": prediction})
    write_jsonl(args.output, output_records)
    print(f"Wrote {len(output_records)} predictions to {args.output}")


if __name__ == "__main__":
    main()
