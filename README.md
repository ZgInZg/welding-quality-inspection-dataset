# VPR-Weld

Code and public data subset for VPR-Weld.

## Contents

- `positive/` and `negative/`: 30 public welding samples with defect annotations and welding parameters.
- `prompts/`: Prompt templates corresponding to Appendix Listings A1-A5.
- `scripts/`: Retrieval, inference, and evaluation scripts.
- `requirements.txt`: Python dependencies.

The complete experimental dataset contains 410 samples. The remaining industrial images and robot-specific production records are not publicly distributed because of industrial data restrictions.

## Installation

```bash
conda create -n vpr-weld python=3.10 -y
conda activate vpr-weld
pip install -r requirements.txt
```

For inference, copy `.env.example` to `.env` and configure the model endpoint and API key. No credential is included in this repository.

## Usage

Each script provides its input arguments through `--help`. Typical commands are:

```bash
# CLIP retrieval
python scripts/run_retrieval.py --data-root . --output retrieval.csv --top-k 5

# Retrieval voting metrics
python scripts/evaluate_retrieval.py --input retrieval.csv --output retrieval_metrics.csv --top-k 5 --voting weighted

# Defect-detection metrics
python scripts/evaluate_detection.py --input predictions.csv --output detection_metrics.csv

# BERTScore with RoBERTa-large
python scripts/evaluate_bertscore.py --input diagnosis_pairs.csv --output-prefix bertscore

# LLM-as-a-judge aggregation
python scripts/summarize_judge_scores.py --input judge_scores.csv --output-prefix judge

# Human-expert alignment
python scripts/evaluate_human_alignment.py --input human_scores.csv --output-prefix alignment

# Safety-veto success rate
python scripts/evaluate_svsr.py --input safety_veto_cases.csv --output-prefix svsr
```

Inference scripts:

```bash
python scripts/run_diagnosis.py --input diagnosis_tasks.jsonl --output diagnosis_predictions.jsonl
python scripts/run_optimization.py --input optimization_tasks.jsonl --output optimization_predictions.jsonl
python scripts/run_llm_judge.py --task diagnosis --input diagnosis_predictions.jsonl --output diagnosis_scores.jsonl
```

The scripts export CSV or JSONL results. Figure-generation and manuscript-formatting code is not included.
