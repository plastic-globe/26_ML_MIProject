"""Convert Hugging Face cais/mmlu into the local unified CSV or JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset


EXPECTED_OUTPUT_FIELDS = [
    "question",
    "A",
    "B",
    "C",
    "D",
    "answer",
    "subject",
    "split",
    "source_file",
]

VALID_SPLITS = ("auxiliary_train", "dev", "val", "validation", "test")
ANSWER_LETTERS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class ConvertedRow:
    question: str
    choice_a: str
    choice_b: str
    choice_c: str
    choice_d: str
    answer: str
    subject: str
    split: str
    source_file: str

    def as_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "A": self.choice_a,
            "B": self.choice_b,
            "C": self.choice_c,
            "D": self.choice_d,
            "answer": self.answer,
            "subject": self.subject,
            "split": self.split,
            "source_file": self.source_file,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination .csv or .jsonl file for the merged dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="Splits to include. Examples: test dev val auxiliary_train",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject whitelist, e.g. abstract_algebra anatomy",
    )
    parser.add_argument(
        "--config",
        default="all",
        help="Hugging Face config name for cais/mmlu. Use 'all' unless you need a specific subject config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    requested_splits = normalize_splits(args.splits)
    subject_filter = set(args.subjects) if args.subjects else None

    rows = collect_rows_from_hf(
        config_name=args.config,
        splits=requested_splits,
        subjects=subject_filter,
    )
    if not rows:
        raise ValueError(
            f"No rows were collected from cais/mmlu for splits={requested_splits} "
            f"and subjects={sorted(subject_filter) if subject_filter else 'ALL'}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        write_csv(output_path, rows)
    elif output_path.suffix.lower() == ".jsonl":
        write_jsonl(output_path, rows)
    else:
        raise ValueError("output-path must end with .csv or .jsonl")

    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"from Hugging Face cais/mmlu config={args.config} splits={requested_splits}"
    )


def normalize_splits(splits: list[str]) -> list[str]:
    normalized = []
    for split in splits:
        canonical = "val" if split == "validation" else split
        if split not in VALID_SPLITS and canonical not in VALID_SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        normalized.append(canonical)
    return normalized


def collect_rows_from_hf(
    *,
    config_name: str,
    splits: list[str],
    subjects: set[str] | None,
) -> list[ConvertedRow]:
    rows: list[ConvertedRow] = []
    for split in splits:
        dataset = load_dataset("cais/mmlu", config_name, split=split)
        for sample in dataset:
            subject = str(sample.get("subject", config_name)).strip()
            if subjects is not None and subject not in subjects:
                continue
            choices = sample["choices"]
            if len(choices) != 4:
                raise ValueError(f"Expected 4 choices, got {len(choices)} for subject={subject}")
            answer = normalize_answer(sample["answer"])
            rows.append(
                ConvertedRow(
                    question=str(sample["question"]).strip(),
                    choice_a=str(choices[0]).strip(),
                    choice_b=str(choices[1]).strip(),
                    choice_c=str(choices[2]).strip(),
                    choice_d=str(choices[3]).strip(),
                    answer=answer,
                    subject=subject,
                    split=split,
                    source_file=f"hf://cais/mmlu/{config_name}/{split}",
                )
            )
    return rows


def normalize_answer(value) -> str:
    if isinstance(value, int):
        if value < 0 or value >= len(ANSWER_LETTERS):
            raise ValueError(f"Invalid integer answer index: {value}")
        return ANSWER_LETTERS[value]

    text = str(value).strip().upper()
    if text in ANSWER_LETTERS:
        return text
    if text.isdigit():
        index = int(text)
        if index < 0 or index >= len(ANSWER_LETTERS):
            raise ValueError(f"Invalid numeric answer index: {text}")
        return ANSWER_LETTERS[index]
    raise ValueError(f"Unsupported answer format: {value!r}")


def write_csv(path: Path, rows: list[ConvertedRow]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPECTED_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)


def write_jsonl(path: Path, rows: list[ConvertedRow]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
