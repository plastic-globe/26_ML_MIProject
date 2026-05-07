"""Run behavior evaluation for sycophancy-style MMLU prompts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from mi_repro.data import build_prompt_examples, load_mmlu_records
from mi_repro.metrics import BehaviorOutcome
from mi_repro.modeling import generate_single_token_choice, load_model
from mi_repro.prompts import ANSWER_LETTERS, build_condition_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--data-path", required=True, help="Path to MMLU-style .csv or .jsonl file")
    parser.add_argument("--output-dir", default="results/behavior", help="Directory for metrics and predictions")
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for wrong-answer assignment")
    parser.add_argument("--device", default="auto", help="Transformers device_map argument")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["plain", "opinion_only", "first_person", "third_person"],
        choices=["plain", "opinion_only", "first_person", "third_person"],
    )
    parser.add_argument(
        "--expertise-levels",
        nargs="+",
        default=["beginner", "intermediate", "advanced"],
        choices=["beginner", "intermediate", "advanced"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_mmlu_records(args.data_path, limit=args.limit)
    examples = build_prompt_examples(records, seed=args.seed)
    loaded = load_model(args.model, device=args.device, dtype=args.dtype)

    prediction_rows: list[dict[str, object]] = []
    metric_buckets: dict[tuple[str, str], list[BehaviorOutcome]] = defaultdict(list)

    for example_index, example in enumerate(examples):
        for condition in args.conditions:
            levels = args.expertise_levels if condition in {"first_person", "third_person"} else [""]
            for level in levels:
                prompt = build_condition_prompt(
                    condition,
                    example,
                    expertise_level=level or None,
                    template_index=example_index,
                )
                raw_prediction = generate_single_token_choice(loaded, prompt)
                normalized_prediction = normalize_answer(raw_prediction)
                outcome = BehaviorOutcome(
                    prediction=normalized_prediction,
                    correct_answer=example.correct_answer,
                    incorrect_answer=example.incorrect_answer,
                )
                metric_buckets[(condition, level)].append(outcome)
                prediction_rows.append(
                    {
                        "example_index": example_index,
                        "subject": example.subject,
                        "condition": condition,
                        "expertise_level": level,
                        "correct_answer": example.correct_answer,
                        "incorrect_answer": example.incorrect_answer,
                        "prediction_raw": raw_prediction,
                        "prediction": normalized_prediction,
                        "prompt": prompt,
                    }
                )

    metrics_rows = build_metrics_rows(metric_buckets)
    write_predictions(output_dir / "predictions.csv", prediction_rows)
    write_metrics(output_dir / "metrics.csv", metrics_rows)
    (output_dir / "metrics.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")


def normalize_answer(text: str) -> str:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else "INVALID"


def build_metrics_rows(metric_buckets: dict[tuple[str, str], list[BehaviorOutcome]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (condition, level), outcomes in sorted(metric_buckets.items()):
        total = len(outcomes)
        rows.append(
            {
                "condition": condition,
                "expertise_level": level,
                "num_samples": total,
                "accuracy": safe_rate(sum(item.is_correct for item in outcomes), total),
                "sycophancy_rate": safe_rate(sum(item.is_sycophantic for item in outcomes), total),
                "independent_error_rate": safe_rate(sum(item.is_independent_error for item in outcomes), total),
                "invalid_rate": safe_rate(sum(item.prediction not in ANSWER_LETTERS for item in outcomes), total),
            }
        )
    return rows


def safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

