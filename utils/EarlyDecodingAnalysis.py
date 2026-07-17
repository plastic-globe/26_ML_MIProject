import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


class EarlyDecodingAnalysis:
    def __init__(self, base_dir="output_inference", save_dir="output_analysis"):
        self.base_dir = Path(base_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.analysis_results = {}

    def _parse_metadata(self, pkl_file):
        relative_parts = pkl_file.relative_to(self.base_dir).parts
        if len(relative_parts) < 3:
            raise ValueError(f"Unexpected mechanistic result path: {pkl_file}")

        dataset = relative_parts[0]
        question_type = relative_parts[1]
        prefix_type = ""
        prefix_subtype = ""
        academic_level = ""

        if question_type == "prefix_and_opinion":
            if len(relative_parts) >= 5:
                prefix_type = relative_parts[2]
                prefix_subtype = relative_parts[3]
                if prefix_type == "academic" and len(relative_parts) >= 6:
                    academic_level = relative_parts[4]
            elif len(relative_parts) >= 4:
                prefix_type = relative_parts[2]
                prefix_subtype = relative_parts[3]

        filename_parts = pkl_file.stem.split("_")
        if len(filename_parts) < 4:
            raise ValueError(f"Unexpected filename format: {pkl_file.name}")

        inference_mode = filename_parts[-3]
        layer_info = filename_parts[-2]
        timestamp = filename_parts[-1]
        model_name = "_".join(filename_parts[:-3])

        return {
            "dataset": dataset,
            "question_type": question_type,
            "prefix_type": prefix_type,
            "prefix_subtype": prefix_subtype,
            "academic_level": academic_level,
            "model_name": model_name,
            "inference_mode": inference_mode,
            "layer_info": layer_info,
            "timestamp": timestamp,
            "file_path": str(pkl_file),
        }

    def load_results(self):
        print("Loading mechanistic result files...")
        pkl_files = sorted(self.base_dir.glob("**/*.pkl"))

        for pkl_file in pkl_files:
            try:
                metadata = self._parse_metadata(pkl_file)
                df = pd.read_pickle(pkl_file)
                key = (
                    f"{metadata['model_name']}|{metadata['question_type']}|"
                    f"{metadata['prefix_type']}|{metadata['prefix_subtype']}|"
                    f"{metadata['academic_level']}|{metadata['inference_mode']}|{metadata['layer_info']}"
                )
                self.results[key] = {
                    "data": df,
                    **metadata,
                }
                print(f"Loaded: {key} ({len(df)} questions)")
            except Exception as exc:
                print(f"Error loading {pkl_file}: {exc}")

        print(f"Total loaded: {len(self.results)} result sets")
        return self.results

    def extract_layer_predictions(self, layer_logits_dict):
        layer_predictions = {}

        for layer_name, logits in layer_logits_dict.items():
            if not isinstance(logits, dict) or not all(letter in logits for letter in "ABCD"):
                continue

            logit_values = np.array([logits[letter] for letter in "ABCD"], dtype=np.float64)
            shifted = logit_values - np.max(logit_values)
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            prediction = "ABCD"[int(np.argmax(probs))]

            layer_predictions[layer_name] = {
                "prediction": prediction,
                "confidence": float(np.max(probs)),
                "logits": logits,
                "probabilities": dict(zip("ABCD", probs.tolist())),
            }

        return layer_predictions

    def _get_sorted_layer_numbers(self, layer_preds):
        layer_numbers = []
        for layer_name in layer_preds.keys():
            try:
                layer_numbers.append(int(layer_name.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return sorted(layer_numbers)

    def analyze_early_decoding_patterns(self):
        print("Analyzing early decoding patterns...")

        for key, result_data in self.results.items():
            df = result_data["data"]
            print(f"\nAnalyzing {key}...")
            layer_analysis = []

            for row_idx, (_, row) in enumerate(df.iterrows()):
                layer_logits = row.get("layer_logits")
                if not isinstance(layer_logits, dict) or not layer_logits:
                    continue

                layer_preds = self.extract_layer_predictions(layer_logits)
                layer_nums = self._get_sorted_layer_numbers(layer_preds)
                if not layer_nums:
                    continue

                predictions_by_layer = []
                confidences_by_layer = []
                for layer_num in layer_nums:
                    layer_key = f"layer_{layer_num}"
                    if layer_key in layer_preds:
                        predictions_by_layer.append(layer_preds[layer_key]["prediction"])
                        confidences_by_layer.append(layer_preds[layer_key]["confidence"])

                if not predictions_by_layer:
                    continue

                final_prediction = predictions_by_layer[-1]
                stabilization_layer = None
                for i in range(len(predictions_by_layer)):
                    if predictions_by_layer[i] != final_prediction:
                        continue

                    trailing = predictions_by_layer[i:]
                    if trailing and all(pred == final_prediction for pred in trailing):
                        stabilization_layer = layer_nums[i]
                        break

                layer_analysis.append(
                    {
                        "question_idx": row_idx,
                        "uid": row.get("uid", row_idx),
                        "final_prediction": final_prediction,
                        "correct_answer": row.get("answer"),
                        "opinion_answer": row.get("opinion", row.get("human_opinion")),
                        "stabilization_layer": stabilization_layer,
                        "predictions_by_layer": predictions_by_layer,
                        "confidences_by_layer": confidences_by_layer,
                        "layer_numbers": layer_nums,
                        "is_correct": final_prediction == row.get("answer") if row.get("answer") else None,
                    }
                )

            self.analysis_results[key] = {
                "layer_analysis": layer_analysis,
                "model_info": result_data,
            }

            stabilization_layers = [
                item["stabilization_layer"]
                for item in layer_analysis
                if item["stabilization_layer"] is not None
            ]
            if stabilization_layers:
                print(f"  Average stabilization layer: {np.mean(stabilization_layers):.2f}")
                print(f"  Median stabilization layer: {np.median(stabilization_layers):.2f}")
                print(
                    f"  Questions that stabilize: {len(stabilization_layers)}/{len(layer_analysis)} "
                    f"({len(stabilization_layers) / len(layer_analysis) * 100:.1f}%)"
                )

    def compare_plain_vs_opinion(self):
        print("\nComparing plain vs opinion-only questions...")
        plain_keys = [
            key
            for key, value in self.analysis_results.items()
            if value["model_info"]["question_type"] == "plain"
        ]
        opinion_keys = [
            key
            for key, value in self.analysis_results.items()
            if value["model_info"]["question_type"] == "opinion_only"
        ]

        comparison_results = {}
        for plain_key in plain_keys:
            plain_meta = self.analysis_results[plain_key]["model_info"]
            matched_opinion_key = None
            for opinion_key in opinion_keys:
                opinion_meta = self.analysis_results[opinion_key]["model_info"]
                if (
                    plain_meta["model_name"] == opinion_meta["model_name"]
                    and plain_meta["inference_mode"] == opinion_meta["inference_mode"]
                    and plain_meta["layer_info"] == opinion_meta["layer_info"]
                ):
                    matched_opinion_key = opinion_key
                    break

            if not matched_opinion_key:
                continue

            plain_analysis = self.analysis_results[plain_key]["layer_analysis"]
            opinion_analysis = self.analysis_results[matched_opinion_key]["layer_analysis"]
            plain_stab = [x["stabilization_layer"] for x in plain_analysis if x["stabilization_layer"] is not None]
            opinion_stab = [x["stabilization_layer"] for x in opinion_analysis if x["stabilization_layer"] is not None]

            if not plain_stab or not opinion_stab:
                continue

            stat, p_value = stats.mannwhitneyu(plain_stab, opinion_stab, alternative="two-sided")
            effect_size = (
                (np.mean(opinion_stab) - np.mean(plain_stab))
                / np.std(plain_stab + opinion_stab)
                if np.std(plain_stab + opinion_stab) > 0
                else 0.0
            )

            model_key = f"{plain_meta['model_name']}|{plain_meta['inference_mode']}|{plain_meta['layer_info']}"
            comparison_results[model_key] = {
                "plain_mean_stab": float(np.mean(plain_stab)),
                "plain_median_stab": float(np.median(plain_stab)),
                "opinion_mean_stab": float(np.mean(opinion_stab)),
                "opinion_median_stab": float(np.median(opinion_stab)),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
                "plain_count": len(plain_stab),
                "opinion_count": len(opinion_stab),
            }

        return comparison_results

    def analyze_sycophancy_emergence(self):
        print("\nAnalyzing sycophancy emergence...")
        for key, analysis_bundle in self.analysis_results.items():
            if analysis_bundle["model_info"]["question_type"] not in {"opinion_only", "prefix_and_opinion"}:
                continue

            sycophancy_analysis = []
            for item in analysis_bundle["layer_analysis"]:
                opinion_answer = item.get("opinion_answer")
                if opinion_answer not in {"A", "B", "C", "D"}:
                    continue

                alignments = [pred == opinion_answer for pred in item["predictions_by_layer"]]
                sycophancy_start_layer = None
                for idx, aligned in enumerate(alignments):
                    if aligned and all(alignments[idx:]):
                        sycophancy_start_layer = item["layer_numbers"][idx]
                        break

                sycophancy_analysis.append(
                    {
                        "question_idx": item["question_idx"],
                        "uid": item["uid"],
                        "opinion_answer": opinion_answer,
                        "correct_answer": item.get("correct_answer"),
                        "final_prediction": item["final_prediction"],
                        "sycophancy_start_layer": sycophancy_start_layer,
                        "is_sycophantic": item["final_prediction"] == opinion_answer,
                        "is_correct": item.get("is_correct"),
                        "layer_numbers": item["layer_numbers"],
                        "opinion_alignment_by_layer": alignments,
                    }
                )

            if sycophancy_analysis:
                analysis_bundle["sycophancy_analysis"] = sycophancy_analysis
                sycophantic_items = [x for x in sycophancy_analysis if x["is_sycophantic"]]
                start_layers = [
                    x["sycophancy_start_layer"]
                    for x in sycophantic_items
                    if x["sycophancy_start_layer"] is not None
                ]
                print(f"\nAnalyzing sycophancy in {key}")
                print(
                    f"  Sycophantic responses: {len(sycophantic_items)}/{len(sycophancy_analysis)} "
                    f"({len(sycophantic_items) / len(sycophancy_analysis) * 100:.1f}%)"
                )
                if start_layers:
                    print(f"  Average sycophancy emergence layer: {np.mean(start_layers):.2f}")
                    print(f"  Median sycophancy emergence layer: {np.median(start_layers):.2f}")

    def export_summary_tables(self):
        rows = []
        for key, analysis_bundle in self.analysis_results.items():
            meta = analysis_bundle["model_info"]
            layer_analysis = analysis_bundle["layer_analysis"]
            stabilization_layers = [
                item["stabilization_layer"]
                for item in layer_analysis
                if item["stabilization_layer"] is not None
            ]
            valid_correct = [item["is_correct"] for item in layer_analysis if item["is_correct"] is not None]

            row = {
                "key": key,
                "dataset": meta["dataset"],
                "model_name": meta["model_name"],
                "question_type": meta["question_type"],
                "prefix_type": meta["prefix_type"],
                "prefix_subtype": meta["prefix_subtype"],
                "academic_level": meta["academic_level"],
                "inference_mode": meta["inference_mode"],
                "layer_info": meta["layer_info"],
                "n_questions": len(layer_analysis),
                "n_stabilized": len(stabilization_layers),
                "stabilization_rate": len(stabilization_layers) / len(layer_analysis) if layer_analysis else np.nan,
                "mean_stabilization_layer": float(np.mean(stabilization_layers)) if stabilization_layers else np.nan,
                "median_stabilization_layer": float(np.median(stabilization_layers)) if stabilization_layers else np.nan,
                "accuracy": float(np.mean(valid_correct)) if valid_correct else np.nan,
                "file_path": meta["file_path"],
            }

            sycophancy_analysis = analysis_bundle.get("sycophancy_analysis", [])
            if sycophancy_analysis:
                sycophantic_items = [x for x in sycophancy_analysis if x["is_sycophantic"]]
                start_layers = [
                    x["sycophancy_start_layer"]
                    for x in sycophantic_items
                    if x["sycophancy_start_layer"] is not None
                ]
                row["sycophancy_rate"] = len(sycophantic_items) / len(sycophancy_analysis)
                row["mean_sycophancy_start_layer"] = float(np.mean(start_layers)) if start_layers else np.nan
                row["median_sycophancy_start_layer"] = float(np.median(start_layers)) if start_layers else np.nan
            else:
                row["sycophancy_rate"] = np.nan
                row["mean_sycophancy_start_layer"] = np.nan
                row["median_sycophancy_start_layer"] = np.nan

            rows.append(row)

        summary_df = pd.DataFrame(rows)
        summary_path = self.save_dir / "mechanistic_runs_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved summary table to {summary_path}")

        comparison_results = self.compare_plain_vs_opinion()
        comparison_columns = [
            "comparison_key",
            "plain_mean_stab",
            "plain_median_stab",
            "opinion_mean_stab",
            "opinion_median_stab",
            "p_value",
            "effect_size",
            "plain_count",
            "opinion_count",
        ]
        comparison_df = pd.DataFrame(
            [
                {"comparison_key": key, **value}
                for key, value in comparison_results.items()
            ],
            columns=comparison_columns,
        )
        comparison_path = self.save_dir / "mechanistic_plain_vs_opinion_summary.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        print(f"Saved comparison table to {comparison_path}")

        return summary_df, comparison_df

    def create_visualizations(self):
        if not self.analysis_results:
            print("No analysis results available for plotting.")
            return

        plt.style.use("default")
        sns.set_palette("deep")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Mechanistic Analysis Summary", fontsize=16, fontweight="bold")

        ax1 = axes[0, 0]
        stabilization_rows = []
        for analysis_bundle in self.analysis_results.values():
            meta = analysis_bundle["model_info"]
            for item in analysis_bundle["layer_analysis"]:
                if item["stabilization_layer"] is None:
                    continue
                stabilization_rows.append(
                    {
                        "Model": meta["model_name"],
                        "Question Type": meta["question_type"],
                        "Stabilization Layer": item["stabilization_layer"],
                    }
                )
        if stabilization_rows:
            stab_df = pd.DataFrame(stabilization_rows)
            sns.boxplot(
                data=stab_df,
                x="Question Type",
                y="Stabilization Layer",
                hue="Model",
                ax=ax1,
            )
            ax1.set_title("Stabilization Layer Distribution")
            ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax1.text(0.5, 0.5, "No stabilization data", ha="center", va="center")

        ax2 = axes[0, 1]
        for analysis_bundle in self.analysis_results.values():
            meta = analysis_bundle["model_info"]
            layer_confidences = {}
            for item in analysis_bundle["layer_analysis"]:
                for layer_num, conf in zip(item["layer_numbers"], item["confidences_by_layer"]):
                    layer_confidences.setdefault(layer_num, []).append(conf)
            if layer_confidences:
                layers = sorted(layer_confidences.keys())
                mean_confs = [np.mean(layer_confidences[layer]) for layer in layers]
                ax2.plot(
                    layers,
                    mean_confs,
                    marker="o",
                    label=f"{meta['model_name']}|{meta['question_type']}",
                )
        ax2.set_title("Average Confidence Across Layers")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Confidence")
        ax2.grid(True, alpha=0.3)
        if ax2.lines:
            ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")

        ax3 = axes[1, 0]
        comparison_results = self.compare_plain_vs_opinion()
        if comparison_results:
            comparison_df = pd.DataFrame(
                [
                    {
                        "comparison_key": key,
                        "plain_mean_stab": value["plain_mean_stab"],
                        "opinion_mean_stab": value["opinion_mean_stab"],
                    }
                    for key, value in comparison_results.items()
                ]
            )
            x = np.arange(len(comparison_df))
            width = 0.35
            ax3.bar(x - width / 2, comparison_df["plain_mean_stab"], width, label="Plain")
            ax3.bar(x + width / 2, comparison_df["opinion_mean_stab"], width, label="Opinion")
            ax3.set_xticks(x)
            ax3.set_xticklabels(comparison_df["comparison_key"], rotation=30, ha="right")
            ax3.set_title("Mean Stabilization Layer: Plain vs Opinion")
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, "No plain vs opinion pairs", ha="center", va="center")

        ax4 = axes[1, 1]
        plotted = False
        for analysis_bundle in self.analysis_results.values():
            meta = analysis_bundle["model_info"]
            sycophancy_analysis = analysis_bundle.get("sycophancy_analysis", [])
            if not sycophancy_analysis:
                continue
            max_layer = max(max(item["layer_numbers"]) for item in sycophancy_analysis if item["layer_numbers"])
            layer_rates = []
            layers = []
            for layer in range(max_layer + 1):
                valid_items = [
                    item["opinion_alignment_by_layer"][item["layer_numbers"].index(layer)]
                    for item in sycophancy_analysis
                    if layer in item["layer_numbers"]
                ]
                if valid_items:
                    layers.append(layer)
                    layer_rates.append(float(np.mean(valid_items)))
            if layers:
                ax4.plot(layers, layer_rates, marker="o", label=f"{meta['model_name']}|{meta['question_type']}")
                plotted = True
        ax4.set_title("Sycophancy Rate Across Layers")
        ax4.set_xlabel("Layer")
        ax4.set_ylabel("Rate")
        ax4.grid(True, alpha=0.3)
        if plotted:
            ax4.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax4.text(0.5, 0.5, "No sycophancy data", ha="center", va="center")

        plt.tight_layout()
        plot_path = self.save_dir / "mechanistic_analysis_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

    def generate_summary_report(self):
        print("\n" + "=" * 60)
        print("MECHANISTIC ANALYSIS SUMMARY REPORT")
        print("=" * 60)

        for key, analysis_bundle in self.analysis_results.items():
            meta = analysis_bundle["model_info"]
            layer_analysis = analysis_bundle["layer_analysis"]
            print(f"\n{key}")
            print("-" * len(key))
            print(f"Dataset: {meta['dataset']}")
            print(f"Questions analyzed: {len(layer_analysis)}")

            stabilization_layers = [
                item["stabilization_layer"]
                for item in layer_analysis
                if item["stabilization_layer"] is not None
            ]
            if stabilization_layers:
                print(f"Mean stabilization layer: {np.mean(stabilization_layers):.2f}")
                print(f"Median stabilization layer: {np.median(stabilization_layers):.2f}")
                print(
                    f"Stabilization rate: {len(stabilization_layers) / len(layer_analysis) * 100:.1f}%"
                )

            valid_correct = [item["is_correct"] for item in layer_analysis if item["is_correct"] is not None]
            if valid_correct:
                print(f"Accuracy: {np.mean(valid_correct):.3f}")

            sycophancy_analysis = analysis_bundle.get("sycophancy_analysis", [])
            if sycophancy_analysis:
                sycophantic_items = [x for x in sycophancy_analysis if x["is_sycophantic"]]
                print(
                    f"Sycophantic responses: {len(sycophantic_items)}/{len(sycophancy_analysis)} "
                    f"({len(sycophantic_items) / len(sycophancy_analysis) * 100:.1f}%)"
                )

        print("\n" + "=" * 60)
        print("Analysis complete.")
        print("=" * 60)


def main():
    analyzer = EarlyDecodingAnalysis()
    analyzer.load_results()

    if not analyzer.results:
        print("No result files found. Please check the directory structure.")
        return analyzer

    analyzer.analyze_early_decoding_patterns()
    analyzer.analyze_sycophancy_emergence()
    analyzer.export_summary_tables()
    analyzer.create_visualizations()
    analyzer.generate_summary_report()
    return analyzer


if __name__ == "__main__":
    main()
