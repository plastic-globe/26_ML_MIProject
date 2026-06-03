import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep activation steering parameters and rank the best mitigation settings."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name used for the sweep.")
    parser.add_argument(
        "--plain_filename",
        type=str,
        default="../../lib/plain/mmlu_plain.pkl",
        help="Plain dataset for steering vector learning.",
    )
    parser.add_argument(
        "--opinion_filename",
        type=str,
        default="../../lib/opinion_only/prefix/mmlu_opinion_only.pkl",
        help="Opinion-only dataset for evaluation.",
    )
    parser.add_argument(
        "--layer_indices",
        type=str,
        default="12,16,20",
        help="Comma-separated list of layer indices to try.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="1.0,2.0,4.0",
        help="Comma-separated list of alpha values to try.",
    )
    parser.add_argument(
        "--train_sizes",
        type=str,
        default="128,512",
        help="Comma-separated list of train sizes to try.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=500,
        help="Rows evaluated per configuration.",
    )
    parser.add_argument(
        "--vector_direction",
        type=str,
        default="restore_plain",
        choices=["restore_plain", "amplify_opinion"],
        help="Steering vector direction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output_improvement",
        help="Root directory for sweep outputs.",
    )
    return parser.parse_args()


def parse_csv_numbers(text, cast_type):
    return [cast_type(item.strip()) for item in text.split(",") if item.strip()]


def run_configuration(args, layer_index, alpha, train_size):
    script_path = Path(__file__).resolve().parents[1] / "steering_analysis" / "run_activation_steering.py"
    python_executable = os.environ.get("PYTHON_EXECUTABLE", "python")

    command = [
        python_executable,
        str(script_path),
        "--model_name",
        args.model_name,
        "--plain_filename",
        args.plain_filename,
        "--opinion_filename",
        args.opinion_filename,
        "--layer_index",
        str(layer_index),
        "--alpha",
        str(alpha),
        "--train_size",
        str(train_size),
        "--eval_size",
        str(args.eval_size),
        "--vector_direction",
        args.vector_direction,
        "--seed",
        str(args.seed),
        "--output_root",
        args.output_root,
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Steering run failed for layer={layer_index}, alpha={alpha}, train_size={train_size}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    metrics = None
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("METRICS_JSON::"):
            metrics = json.loads(stripped.split("METRICS_JSON::", 1)[1])
    if metrics is None:
        raise RuntimeError(f"Could not parse metrics from steering run output:\n{result.stdout}")

    metrics["sweep_layer_index"] = layer_index
    metrics["sweep_alpha"] = alpha
    metrics["sweep_train_size"] = train_size
    return metrics


def main():
    args = parse_args()
    layer_indices = parse_csv_numbers(args.layer_indices, int)
    alphas = parse_csv_numbers(args.alphas, float)
    train_sizes = parse_csv_numbers(args.train_sizes, int)

    all_metrics = []
    for layer_index in layer_indices:
        for alpha in alphas:
            for train_size in train_sizes:
                print(f"Running layer={layer_index}, alpha={alpha}, train_size={train_size}")
                metrics = run_configuration(args, layer_index, alpha, train_size)
                all_metrics.append(metrics)

    summary_df = pd.DataFrame(all_metrics)
    summary_df = summary_df.sort_values(
        by=["sycophancy_delta", "accuracy_delta"],
        ascending=[True, False],
    ).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split("/")[-1].replace(".", "_")
    output_dir = Path(args.output_root) / "mmlu" / "steering_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_short_name}_{timestamp}.csv"
    summary_df.to_csv(output_path, index=False)

    print(summary_df.head(10).to_string(index=False))
    print(f"Saved sweep summary to {output_path}")


if __name__ == "__main__":
    main()
