from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", default=r"D:\26_ML_MIProject\lib\plain\mmlu_plain.pkl")
    parser.add_argument("--opinion", default=r"D:\26_ML_MIProject\lib\opinion_only\prefix\mmlu_opinion_only.pkl")
    parser.add_argument("--output", default=r"D:\MI_project\outputs\qwen_mmlu_raw_full.csv")
    args = parser.parse_args()

    plain = pd.read_pickle(args.plain)
    opinion = pd.read_pickle(args.opinion)[["uid", "opinion"]].copy()
    merged = plain.merge(opinion, on="uid", how="inner", validate="one_to_one")
    columns = ["uid", "question", "subject", "choices", "A", "B", "C", "D", "answer", "full_question", "opinion"]
    merged = merged[columns].copy()
    merged["answer"] = merged["answer"].astype(str).str.strip().str.upper()
    merged["opinion"] = merged["opinion"].astype(str).str.strip().str.upper()
    bad = merged[merged["answer"] == merged["opinion"]]
    if not bad.empty:
        raise RuntimeError(f"Found {len(bad)} rows where opinion equals answer")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"wrote {out} rows={len(merged)} subjects={merged['subject'].nunique()}")


if __name__ == "__main__":
    main()
