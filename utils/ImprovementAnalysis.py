from pathlib import Path

import pandas as pd


class ImprovementAnalysis:
    def __init__(self, base_dir="output_improvement"):
        self.base_dir = Path(base_dir)

    def load_prompt_runs(self):
        files = list((self.base_dir / "mmlu" / "prompt_mitigation").glob("*.json"))
        rows = [pd.read_json(file, typ="series").to_dict() for file in files]
        return pd.DataFrame(rows)

    def load_steering_sweeps(self):
        files = list((self.base_dir / "mmlu" / "steering_sweep").glob("*.csv"))
        frames = [pd.read_csv(file) for file in files]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def summarize(self):
        prompt_df = self.load_prompt_runs()
        steering_df = self.load_steering_sweeps()

        if not prompt_df.empty:
            prompt_best = prompt_df.sort_values(
                by=["sycophancy_delta", "accuracy_delta"],
                ascending=[True, False],
            ).head(10)
            print("Best prompt mitigation runs:")
            print(prompt_best.to_string(index=False))

        if not steering_df.empty:
            steering_best = steering_df.sort_values(
                by=["sycophancy_delta", "accuracy_delta"],
                ascending=[True, False],
            ).head(10)
            print("\nBest steering sweep runs:")
            print(steering_best.to_string(index=False))

        if prompt_df.empty and steering_df.empty:
            print("No improvement runs found.")


def main():
    ImprovementAnalysis().summarize()


if __name__ == "__main__":
    main()
