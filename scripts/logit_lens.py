"""Export layer-wise answer logits and decision scores for sycophancy analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from mi_repro.data import build_prompt_examples, load_mmlu_records
from mi_repro.logit_lens_utils import add_decision_scores, decode_answer_logits_from_hidden_states
from mi_repro.modeling import answer_choice_logits, load_model, token_id_for_single_letter
from mi_repro.prompts import ANSWER_LETTERS, build_condition_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--data-path", required=True, help="Path to MMLU-style .csv or .jsonl file")
    parser.add_argument("--output-dir", default="results/logit_lens", help="Directory for exported layer-wise data")
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to analyze")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for wrong-answer assignment")
    parser.add_argument("--device", default="auto", help="Transformers device_map argument")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument(
        "--condition-pairs",
        nargs="+",
        default=["plain:opinion_only"],
        help="Condition pairs to compare, e.g. plain:opinion_only plain:first_person",
    )
    parser.add_argument(
        "--expertise-level",
        default="advanced",
        choices=["beginner", "intermediate", "advanced"],
        help="Expertise level used when a condition requires it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_mmlu_records(args.data_path, limit=args.limit)
    examples = build_prompt_examples(records, seed=args.seed)
    loaded = load_model(args.model, device=args.device, dtype=args.dtype)
    answer_token_ids = {letter: token_id_for_single_letter(loaded, letter) for letter in ANSWER_LETTERS}

    row_buffer: list[dict[str, object]] = []
    summary: dict[str, object] = {"model": args.model, "pairs": []}

    for pair in args.condition_pairs:
        source_condition, target_condition = pair.split(":", maxsplit=1)
        pair_rows = []
        for example_index, example in enumerate(examples):
            for condition in (source_condition, target_condition):
                prompt = build_condition_prompt(
                    condition,
                    example,
                    expertise_level=args.expertise_level if condition in {"first_person", "third_person"} else None,
                    template_index=example_index,
                )
                outputs = answer_choice_logits(loaded, prompt, output_hidden_states=True)
                layer_rows = decode_answer_logits_from_hidden_states(
                    loaded.model,
                    outputs.hidden_states,
                    answer_token_ids,
                )
                layer_rows = add_decision_scores(
                    layer_rows,
                    correct_answer=example.correct_answer,
                    incorrect_answer=example.incorrect_answer,
                )
                for row in layer_rows:
                    export_row = {
                        "pair": pair,
                        "example_index": example_index,
                        "subject": example.subject,
                        "condition": condition,
                        "correct_answer": example.correct_answer,
                        "incorrect_answer": example.incorrect_answer,
                        **row,
                    }
                    row_buffer.append(export_row)
                    pair_rows.append(export_row)

        summary["pairs"].append(summarize_pair(pair, pair_rows))

    write_csv(output_dir / "layerwise_logits.csv", row_buffer)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize_pair(pair: str, rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["layer"]))
        grouped.setdefault(key, []).append(row)

    layer_summary = []
    for (condition, layer), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        layer_summary.append(
            {
                "pair": pair,
                "condition": condition,
                "layer": layer,
                "mean_correct_decision_score": average(float(row["correct_decision_score"]) for row in bucket),
                "mean_incorrect_decision_score": average(float(row["incorrect_decision_score"]) for row in bucket),
            }
        )
    return {"pair": pair, "layer_summary": layer_summary}


def average(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
