#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leave-one-out CLIP retrieval on the public subset")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.68)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def inventory(root: Path) -> pd.DataFrame:
    rows = []
    for split, label in (("positive", "OK"), ("negative", "NG")):
        for image_path in sorted((root / split / "images").glob("*.png")):
            rows.append({"sample_id": image_path.stem, "label": label, "image_path": image_path})
    if not rows:
        raise SystemExit(f"No PNG images found under {root}")
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    samples = inventory(args.data_root.resolve())
    if args.top_k >= len(samples):
        raise SystemExit("--top-k must be smaller than the number of samples for leave-one-out retrieval")
    processor = CLIPProcessor.from_pretrained(args.encoder)
    model = CLIPModel.from_pretrained(args.encoder).to(args.device).eval()
    embeddings = []
    for path in tqdm(samples["image_path"], desc="Encoding images"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            feature = model.get_image_features(**inputs)
            feature = feature / feature.norm(dim=-1, keepdim=True)
        embeddings.append(feature.cpu())
    matrix = torch.cat(embeddings, dim=0)
    similarity = matrix @ matrix.T
    similarity.fill_diagonal_(-float("inf"))
    rows = []
    for query_index, query in samples.iterrows():
        values, indices = similarity[query_index].topk(args.top_k)
        rejected = values[0].item() < args.threshold
        for rank, (score_value, reference_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            reference = samples.iloc[reference_index]
            rows.append(
                {
                    "query_id": query["sample_id"],
                    "query_label": query["label"],
                    "reference_id": reference["sample_id"],
                    "reference_label": reference["label"],
                    "rank": rank,
                    "cosine_similarity": score_value,
                    "rejected": rejected,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {len(rows)} retrieval records to {args.output}")


if __name__ == "__main__":
    main()
