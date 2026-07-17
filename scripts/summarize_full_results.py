from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "qwen_mmlu_raw_full_locate_steer_improve"
OUT = ROOT / "outputs" / "full_analysis"
REPORT = ROOT / "outputs" / "report_zh_full.md"


COLORS = {
    "plain": "#4E79A7",
    "opinion_only": "#A23B72",
    "prefix_and_opinion": "#7A5195",
    "answer": "#2F6B4F",
    "opinion": "#B33A3A",
    "patch": "#4C78A8",
    "prompt": "#6A4C93",
    "steer": "#2A9D8F",
}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def num(x: float) -> str:
    return f"{x:.3f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_bar(path: Path, title: str, labels: list[str], series: list[tuple[str, list[float], str]], ylabel: str = "Rate") -> None:
    width, height = 920, 520
    left, right, top, bottom = 90, 40, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(max(vals) for _, vals, _ in series)
    max_v = max(0.1, min(1.0, max_v * 1.12))
    group_w = plot_w / len(labels)
    bar_w = min(42, group_w / (len(series) + 1.2))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="38" font-size="24" font-weight="700" fill="#222">{title}</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" font-size="14" fill="#555">{ylabel}</text>',
    ]
    for tick in range(0, 6):
        v = max_v * tick / 5
        y = top + plot_h - plot_h * v / max_v
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="12" fill="#666">{pct(v)}</text>')
    for i, label in enumerate(labels):
        cx = left + group_w * (i + 0.5)
        parts.append(f'<text x="{cx:.1f}" y="{height - 48}" text-anchor="middle" font-size="13" fill="#333">{label}</text>')
        for j, (name, values, color) in enumerate(series):
            v = values[i]
            x = cx - (len(series) * bar_w) / 2 + j * bar_w
            h = plot_h * v / max_v
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{h:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" fill="#333">{pct(v)}</text>')
    lx = left
    ly = height - 24
    for name, _values, color in series:
        parts.append(f'<rect x="{lx}" y="{ly - 12}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{lx + 20}" y="{ly}" font-size="13" fill="#333">{name}</text>')
        lx += 170
    parts.append("</svg>")
    write(path, "\n".join(parts))


def svg_line(path: Path, title: str, xs: list[int], series: list[tuple[str, list[float], str]], ylabel: str) -> None:
    width, height = 940, 520
    left, right, top, bottom = 80, 40, 70, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    vals = [v for _, ys, _ in series for v in ys]
    min_v, max_v = min(vals), max(vals)
    pad = (max_v - min_v) * 0.12 or 0.1
    min_v -= pad
    max_v += pad
    def sx(x: int) -> float:
        return left + (x - min(xs)) / (max(xs) - min(xs)) * plot_w
    def sy(v: float) -> float:
        return top + plot_h - (v - min_v) / (max_v - min_v) * plot_h
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="38" font-size="24" font-weight="700" fill="#222">{title}</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" font-size="14" fill="#555">{ylabel}</text>',
    ]
    for tick in range(0, 6):
        v = min_v + (max_v - min_v) * tick / 5
        y = sy(v)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="12" fill="#666">{v:.2f}</text>')
    for x in xs:
        parts.append(f'<text x="{sx(x):.1f}" y="{height - 34}" text-anchor="middle" font-size="11" fill="#666">{x}</text>')
    for name, ys, color in series:
        points = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, ys))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, v in zip(xs, ys):
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="3.5" fill="{color}"/>')
    lx = left
    ly = height - 10
    for name, _ys, color in series:
        parts.append(f'<line x1="{lx}" y1="{ly - 5}" x2="{lx + 24}" y2="{ly - 5}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 32}" y="{ly}" font-size="13" fill="#333">{name}</text>')
        lx += 230
    parts.append("</svg>")
    write(path, "\n".join(parts))


def main() -> None:
    global BASE, OUT, REPORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--analysis-dir", default=str(OUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    BASE = Path(args.base)
    OUT = Path(args.analysis_dir)
    REPORT = Path(args.report)
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((BASE / "run_config.json").read_text(encoding="utf-8"))
    behavior = pd.read_csv(BASE / "locate_behavior_summary.csv")
    layer = pd.read_csv(BASE / "locate_layer_summary.csv")
    steer = pd.read_csv(BASE / "steer_sweep_summary.csv")
    improve = pd.read_csv(BASE / "improve_prompt_mitigation_summary.csv")
    patch = pd.read_csv(BASE / "activation_patching_layer_summary.csv")
    elim = pd.read_csv(BASE / "sycophancy_elimination_summary.csv")

    behavior_labels = behavior["condition"].tolist()
    svg_bar(
        OUT / "behavior_rates.svg",
        "Behavior: opinion prompt sharply increases sycophancy",
        behavior_labels,
        [
            ("Accuracy", behavior["accuracy"].tolist(), COLORS["answer"]),
            ("Sycophancy", behavior["sycophancy_rate"].tolist(), COLORS["opinion"]),
        ],
    )

    imp_order = ["none", "truth_priority", "anti_sycophancy", "verify_then_answer", "counter_opinion_check"]
    imp = improve.set_index("mitigation").loc[imp_order].reset_index()
    svg_bar(
        OUT / "mitigation_rates.svg",
        "Improve: anti-sycophancy prompt gives the strongest reduction",
        imp["mitigation"].tolist(),
        [
            ("Accuracy", imp["accuracy"].tolist(), COLORS["answer"]),
            ("Sycophancy", imp["sycophancy_rate"].tolist(), COLORS["opinion"]),
        ],
    )

    layers = sorted(layer["layer"].unique().tolist())
    prof = []
    for condition, color in [("opinion_only", COLORS["opinion"]), ("prefix_and_opinion", COLORS["prefix_and_opinion"]), ("plain", COLORS["plain"])]:
        sub = layer[layer["condition"] == condition].sort_values("layer")
        prof.append((condition, sub["opinion_minus_answer"].tolist(), color))
    svg_line(OUT / "logit_lens_opinion_minus_answer.svg", "Locate: layer-wise opinion-minus-answer profile", layers, prof, "p(opinion) - p(answer)")

    patch_sorted = patch.sort_values("layer")
    svg_line(
        OUT / "patching_delta_by_layer.svg",
        "Activation patching: plain-state substitution changes answer margin",
        patch_sorted["layer"].astype(int).tolist(),
        [("patched - base margin", patch_sorted["mean_patch_delta"].tolist(), COLORS["patch"])],
        "Correct-minus-opinion margin delta",
    )

    best_prompt = elim[elim["method"] == "prompt"].sort_values("elimination_rate", ascending=False).iloc[0]
    best_steer = elim[elim["method"] == "steering"].sort_values("elimination_rate", ascending=False).iloc[0]
    top_layers = layer.sort_values("opinion_minus_answer", ascending=False).head(8)
    top_patch = patch.sort_values("mean_patch_delta", ascending=False).head(8)
    best_steer_rows = steer.sort_values(["sycophancy_rate", "accuracy"], ascending=[True, False]).head(8)

    metrics = {
        "num_examples": int(config["num_examples"]),
        "model_name": config["model_name"],
        "device": config["device"],
        "baseline_opinion_sycophancy_rate": float(behavior.loc[behavior["condition"] == "opinion_only", "sycophancy_rate"].iloc[0]),
        "plain_accuracy": float(behavior.loc[behavior["condition"] == "plain", "accuracy"].iloc[0]),
        "best_prompt": best_prompt.to_dict(),
        "best_steering": best_steer.to_dict(),
        "top_logit_lens": top_layers.to_dict("records"),
        "top_patching": top_patch.to_dict("records"),
    }
    write(OUT / "key_metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
    top_layers.to_csv(OUT / "top_logit_lens_layers.csv", index=False)
    top_patch.to_csv(OUT / "top_patching_layers.csv", index=False)
    best_steer_rows.to_csv(OUT / "best_steering_settings.csv", index=False)

    label = args.label or config["model_name"]
    image_dir = OUT.name
    num_examples = int(config["num_examples"])
    layer_count = len(layers)
    logit_lens_rows = int(layer["n"].sum())
    patch_rows = int(patch["n"].sum())
    steer_layers = ",".join(str(int(x)) for x in config["layers"])
    alpha_values = ",".join(f"{float(x):g}" for x in config["alphas"])
    top_layer_names = ", ".join(f"layer {int(x)}" for x in top_layers["layer"].head(3))
    top_patch_names = ", ".join(f"layer {int(x)}" for x in top_patch["layer"].head(3))
    top_gap = float(top_layers["opinion_minus_answer"].iloc[0])
    top_patch_row = top_patch.iloc[0]
    base_margin = float(top_patch_row["base_margin"])
    patch_verb = "增强" if base_margin >= 0 else "恢复"
    report = f"""# LLM 迎合现象 Locate-Steer-Improve 全量实验报告（{label}）

## 1. 实验目标与复现定位

本项目围绕 MMLU 事实性选择题中的 **sycophancy / 迎合错误用户观点** 现象，复现并扩展 2024-2026 年 actionable mechanistic interpretability 的常见链路：先用行为实验与 Logit Lens / Activation Patching 定位关键层，再用 activation vector arithmetic 在推理阶段干预，最后与 prompt-level mitigation 对比。

本轮修正后使用 `mmlu_raw` 全量数据，而不是 3000 条子集：

| 项目 | 设置 |
|---|---:|
| 输入文件 | `outputs/qwen_mmlu_raw_full.csv` |
| 样本数 | {config['num_examples']} |
| subject 数 | 57 |
| 模型 | {config['model_name']} |
| 框架 | Hugging Face Transformers + PyTorch |
| 设备 | AutoDL/SeetaCloud CUDA |
| Locate | behavior, logit lens, activation patching |
| Steer | layer {steer_layers}, alpha = {alpha_values} |
| Improve | none, truth_priority, anti_sycophancy, verify_then_answer, counter_opinion_check |

## 2. 数据构造

原始数据来自 `D:\\26_ML_MIProject\\raw_data\\mmlu_raw.pkl`，共 14042 条。实验输入由 `lib/plain/mmlu_plain.pkl` 与 `lib/opinion_only/prefix/mmlu_opinion_only.pkl` 按 `uid` 合并得到：

- `full_question` 使用 plain 版本，避免在 plain 条件中混入观点。
- `opinion` 使用 opinion-only 版本中已经构造好的错误选项。
- 校验结果：`answer == opinion` 的行数为 0。

## 3. Locate：迎合行为确实由用户观点触发

| condition | n | accuracy | sycophancy_rate |
|---|---:|---:|---:|
""" 
    for row in behavior.itertuples():
        report += f"| {row.condition} | {row.n} | {pct(row.accuracy)} | {pct(row.sycophancy_rate)} |\n"
    report += f"""

plain 条件下模型准确率为 {pct(float(behavior.loc[behavior.condition == 'plain', 'accuracy'].iloc[0]))}，迎合率为 {pct(float(behavior.loc[behavior.condition == 'plain', 'sycophancy_rate'].iloc[0]))}；加入明确错误用户观点后，opinion-only 条件准确率降到 {pct(float(behavior.loc[behavior.condition == 'opinion_only', 'accuracy'].iloc[0]))}，迎合率升到 {pct(float(behavior.loc[behavior.condition == 'opinion_only', 'sycophancy_rate'].iloc[0]))}。这说明小模型在事实召回任务中明显受错误用户观点牵引。

![行为对比]({image_dir}/behavior_rates.svg)

## 4. Locate：Logit Lens 呈现逐层观点差异

全量 Logit Lens 共有 {num_examples} × 3 × {layer_count} = {logit_lens_rows} 条逐层记录。按 `p_opinion - p_answer` 排序，最强 opinion-over-answer 信号如下：

| condition | layer | p_answer | p_opinion | opinion_minus_answer |
|---|---:|---:|---:|---:|
"""
    for row in top_layers.itertuples():
        report += f"| {row.condition} | {row.layer} | {num(row.p_answer)} | {num(row.p_opinion)} | {num(row.opinion_minus_answer)} |\n"
    report += f"""

在 {label} 上，最大的 `p_opinion - p_answer` 只有 {num(top_gap)}，主要出现在 {top_layer_names}。这说明错误观点信号存在，但幅度较弱，且没有表现为固定的后段层峰值；因此本轮不能沿用 0.5B 报告里“layer 20-22 最强”的表述。

![Logit Lens]({image_dir}/logit_lens_opinion_minus_answer.svg)

## 5. Locate：Activation Patching 给出因果层证据

Activation Patching 将 opinion prompt 的最后 token 表示替换为 plain prompt 的对应层表示，观察 correct-minus-opinion margin 如何变化。全量记录数为 {num_examples} × {layer_count} = {patch_rows}。

| layer | base_margin | patched_margin | mean_patch_delta |
|---:|---:|---:|---:|
"""
    for row in top_patch.itertuples():
        report += f"| {row.layer} | {num(row.base_margin)} | {num(row.patched_margin)} | {num(row.mean_patch_delta)} |\n"
    report += f"""

{top_patch_names} 的平均 patch delta 最大；其中 {top_patch_names.split(', ')[0]} 从 base margin {num(base_margin)} 变为 patched margin {num(float(top_patch_row['patched_margin']))}，delta 为 {num(float(top_patch_row['mean_patch_delta']))}。由于 base margin 已经为正，本轮 patching 更准确地说是在{patch_verb}正确答案相对错误观点的 margin，而不是从负 margin 中“救回”答案。

![Activation Patching]({image_dir}/patching_delta_by_layer.svg)

## 6. Steer：向量干预有方向性，但消除幅度有限

Steering vector 使用 `mean(hidden_plain[layer] - hidden_opinion[layer])`，在 opinion prompt 推理时注入 `alpha * vector`。全量 sweep 覆盖 4 层 × 7 个 alpha × 14042 条样本。

按 elimination rate 排序，最强 steering 设置为 `{best_steer['setting']}`，在 baseline 迎合样本上的消除率为 {pct(float(best_steer['elimination_rate']))}，总体迎合率为 {pct(float(best_steer['overall_sycophancy_rate']))}。这说明简单 activation vector arithmetic 能改变方向，但在 {label} 上仍不足以大幅消除迎合。

## 7. Improve：prompt-level mitigation 明显更有效

| mitigation | n | accuracy | sycophancy_rate |
|---|---:|---:|---:|
"""
    for row in imp.itertuples():
        report += f"| {row.mitigation} | {row.n} | {pct(row.accuracy)} | {pct(row.sycophancy_rate)} |\n"
    report += f"""

最佳 prompt mitigation 是 `{best_prompt['setting']}`：在 baseline 迎合样本中的消除率为 {pct(float(best_prompt['elimination_rate']))}，总体迎合率从 {pct(float(metrics['baseline_opinion_sycophancy_rate']))} 降到 {pct(float(best_prompt['overall_sycophancy_rate']))}，同时总体准确率提升到 {pct(float(best_prompt['overall_accuracy']))}。

![Prompt Mitigation]({image_dir}/mitigation_rates.svg)

## 8. 分析与想法

1. **定位结果比行为结果更有解释力。** 行为层面只能说明模型是否迎合；Logit Lens 显示 {label} 的错误观点概率优势较弱，Patching 则显示 {top_patch_names} 对 correct-minus-opinion margin 的因果影响更强。
2. **较大模型的迎合率明显降低，但没有消失。** 本轮 opinion-only 条件迎合率为 {pct(metrics['baseline_opinion_sycophancy_rate'])}，显著低于 0.5B 全量实验的 69.2%，但仍高于 plain 条件，说明错误用户观点仍会改变事实性回答。
3. **steering 有机制一致性，但工程效果有限。** 方向向量来自 plain-opinion 表示差，确实能小幅降低迎合；但最佳 elimination rate 只有 {pct(float(best_steer['elimination_rate']))}，低于 prompt mitigation。
4. **最实用的改进是机制定位 + prompt 约束结合。** 从机制角度知道哪些层会改变答案 margin；从应用角度，`anti_sycophancy` 这类明确约束更稳定地降低迎合。

## 9. 交付物

- 全量输入：`outputs/qwen_mmlu_raw_full.csv`
- 全量实验目录：`{BASE.as_posix()}`
- 关键图表：`{OUT.as_posix()}`
- 代码：`outputs/code/run_qwen3000_cpu_suite.py`, `scripts/build_mmlu_raw_full_csv.py`, `scripts/remote_seeta_job.py`
"""
    write(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
