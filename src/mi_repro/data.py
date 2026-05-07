"""Dataset helpers for behavior evaluation and layer-wise analysis."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .prompts import ANSWER_LETTERS, PromptExample


@dataclass(frozen=True)
class RawMMLURecord:
    question: str
    choices: tuple[str, str, str, str]
    answer: str
    subject: str


def load_mmlu_records(path: str | Path, *, limit: int | None = None) -> list[RawMMLURecord]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {data_path}. "
            "Provide a local .csv or .jsonl file with MMLU-style rows."
        )

    if data_path.suffix.lower() == ".csv":
        records = list(_load_csv(data_path))
    elif data_path.suffix.lower() == ".jsonl":
        records = list(_load_jsonl(data_path))
    else:
        raise ValueError("Supported dataset formats are .csv and .jsonl")

    return records[:limit] if limit is not None else records


def build_prompt_examples(
    records: Sequence[RawMMLURecord],
    *,
    seed: int,
) -> list[PromptExample]:
    rng = random.Random(seed)
    examples: list[PromptExample] = []
    for record in records:
        wrong_answers = [letter for letter in ANSWER_LETTERS if letter != record.answer]
        incorrect_answer = rng.choice(wrong_answers)
        examples.append(
            PromptExample(
                question=record.question,
                option_a=record.choices[0],
                option_b=record.choices[1],
                option_c=record.choices[2],
                option_d=record.choices[3],
                correct_answer=record.answer,
                incorrect_answer=incorrect_answer,
                subject=record.subject,
            )
        )
    return examples


def _load_csv(path: Path) -> Iterator[RawMMLURecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "A", "B", "C", "D", "answer", "subject"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            yield RawMMLURecord(
                question=row["question"].strip(),
                choices=(row["A"].strip(), row["B"].strip(), row["C"].strip(), row["D"].strip()),
                answer=row["answer"].strip().upper(),
                subject=row["subject"].strip(),
            )


def _load_jsonl(path: Path) -> Iterator[RawMMLURecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                choices = row["choices"]
                answer = row["answer"]
            except KeyError as exc:
                raise ValueError(f"Missing field {exc.args[0]!r} on line {line_number}") from exc
            if isinstance(answer, int):
                answer = ANSWER_LETTERS[answer]
            yield RawMMLURecord(
                question=row["question"].strip(),
                choices=tuple(choice.strip() for choice in choices),
                answer=str(answer).strip().upper(),
                subject=row.get("subject", "unknown").strip(),
            )

