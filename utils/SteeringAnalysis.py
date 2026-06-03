import json
from pathlib import Path

import pandas as pd


class SteeringAnalysis:
    def __init__(self, base_dir="output_steering"):
        self.base_dir = Path(base_dir)
        self.runs = []

    def load_runs(self):
        metric_files = list(self.base_dir.glob("**/*.json"))
        for metric_file in metric_files:
            with open(metric_file, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            self.runs.append(
                {
                    "metrics_path": str(metric_file),
                    **metrics,
                }
            )
        return pd.DataFrame(self.runs)

    def summarize_best_runs(self):
        df = self.load_runs()
        if df.empty:
            print("No steering runs found.")
            return df

        ordered = df.sort_values(
            by=["sycophancy_delta", "accuracy_delta"],
            ascending=[True, False],
        )
        print(ordered.head(10).to_string(index=False))
        return ordered


def main():
    analysis = SteeringAnalysis()
    analysis.summarize_best_runs()


if __name__ == "__main__":
    main()
