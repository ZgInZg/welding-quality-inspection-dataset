from __future__ import annotations

import base64
import json
import math
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def substantive_text(value: Any) -> str:
    def collect(item: Any) -> list[str]:
        if isinstance(item, dict):
            return [text for child in item.values() for text in collect(child)]
        if isinstance(item, list):
            return [text for child in item for text in collect(child)]
        return [str(item).strip()] if item is not None and str(item).strip() else []

    raw = str(value).strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        text = re.sub(r"[{}\[\]\"':,]", " ", raw)
    else:
        text = " ".join(collect(parsed))
    return re.sub(r"\s+", " ", text).strip()


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values]
    return sum(data) / len(data)


def mae(left: Sequence[float], right: Sequence[float]) -> float:
    return mean(abs(float(a) - float(b)) for a, b in zip(left, right))


def pearson_r(left: Sequence[float], right: Sequence[float]) -> float:
    x = [float(value) for value in left]
    y = [float(value) for value in right]
    x_bar, y_bar = mean(x), mean(y)
    numerator = sum((a - x_bar) * (b - y_bar) for a, b in zip(x, y))
    denominator = math.sqrt(sum((a - x_bar) ** 2 for a in x) * sum((b - y_bar) ** 2 for b in y))
    return numerator / denominator if denominator else float("nan")


def round_likert(value: float) -> int:
    return min(max(int(float(value) + 0.5), 1), 5)


def weighted_kappa(left: Sequence[float], right: Sequence[float]) -> float:
    a = [round_likert(value) for value in left]
    b = [round_likert(value) for value in right]
    observed = [[0.0] * 5 for _ in range(5)]
    for first, second in zip(a, b):
        observed[first - 1][second - 1] += 1.0 / len(a)
    left_hist = [sum(row) for row in observed]
    right_hist = [sum(observed[i][j] for i in range(5)) for j in range(5)]
    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(5):
        for j in range(5):
            weight = ((i - j) ** 2) / 16.0
            observed_disagreement += weight * observed[i][j]
            expected_disagreement += weight * left_hist[i] * right_hist[j]
    return 1.0 - observed_disagreement / expected_disagreement if expected_disagreement else 1.0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    pairs = [(int(a), int(b)) for a, b in zip(y_true, y_pred)]
    tp = sum(a == 1 and b == 1 for a, b in pairs)
    tn = sum(a == 0 and b == 0 for a, b in pairs)
    fp = sum(a == 0 and b == 1 for a, b in pairs)
    fn = sum(a == 1 and b == 0 for a, b in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(pairs), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / len(pairs), "precision": precision,
        "recall": recall, "f1": f1,
    }


def image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate(
    prompt: str,
    images: list[str | Path] | None = None,
    env_prefix: str = "VLM",
) -> str:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    base_url = os.getenv(f"{env_prefix}_BASE_URL")
    api_key = os.getenv(f"{env_prefix}_API_KEY") or "local-endpoint"
    model = os.getenv(f"{env_prefix}_MODEL")
    if not model:
        raise RuntimeError(f"{env_prefix}_MODEL is not configured")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for path in images or []:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""
