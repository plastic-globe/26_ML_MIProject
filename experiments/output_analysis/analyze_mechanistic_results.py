import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.EarlyDecodingAnalysis import EarlyDecodingAnalysis


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze mechanistic inference results.")
    parser.add_argument(
        "--input_root",
        type=str,
        default="output_inference",
        help="Root directory containing mechanistic output pickle files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output_analysis",
        help="Directory to save analysis tables and figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyzer = EarlyDecodingAnalysis(base_dir=args.input_root, save_dir=args.save_dir)
    analyzer.load_results()

    if not analyzer.results:
        print(f"No mechanistic result files found under {Path(args.input_root).resolve()}")
        return

    analyzer.analyze_early_decoding_patterns()
    analyzer.analyze_sycophancy_emergence()
    analyzer.export_summary_tables()
    analyzer.create_visualizations()
    analyzer.generate_summary_report()


if __name__ == "__main__":
    main()
