# LLM 迎合现象的 Locate-Steer-Improve 复现实验报告

## 1. 项目目标

本项目围绕大语言模型的 sycophancy（迎合用户错误观点）现象，按照 README 要求完成 `locate -> steer -> improve` 三段实验闭环。实验选择 `Qwen/Qwen2.5-0.5B-Instruct`，在本地 CPU 上运行 100 条 MMLU 样本，覆盖 56 个 subject，避免使用 GPT 系列模型。

课程要求对应关系：

- **Locate**：用 Logit Lens 和 Activation Patching 定位与用户错误观点相关的关键层。
- **Steer & Improve**：用 activation vector arithmetic 在推理阶段干预模型行为，并比较 prompt-level mitigation。
- **顶会/Arxiv 复现**：复现 sycophancy 机制分析论文与 2024-2026 actionable mechanistic interpretability 框架中的核心链路：先定位内部表示，再做推理时干预，最后评估改进策略。
- **交付物**：代码、实验 CSV/PT/JSON 产物、中文报告、中文讲解 PPT。

## 2. 代码与实验入口

主要代码位于：

- `outputs/code/colab_sycophancy_locate_steer_improve.py`：原项目一体化实验脚本，覆盖行为评估、逐层 logit lens、activation steering、prompt mitigation。
- `outputs/code/run_qwen_no_plots.py`：本地环境缺少 matplotlib 时的轻量包装脚本，保留核心 CSV/PT 产物。
- `outputs/code/run_activation_patching_qwen.py`：补充 activation patching 定位实验，在每层 residual stream 上用 plain prompt 表示替换 opinion prompt 表示。

最终数据与结果位于：

- 输入数据：`outputs/qwen_mmlu_100.csv`
- 结果目录：`outputs/qwen_cpu_mmlu100_locate_steer_improve/`

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen/Qwen2.5-0.5B-Instruct |
| 设备 | CPU |
| 框架 | Transformers |
| 数据 | 项目自带 MMLU pkl 构造的 100 条样本，覆盖 56 个 subject |
| 条件 | plain / opinion_only / prefix_and_opinion |
| Steering | layer 23，alpha = -2 / 0 / 2，train_size = 32 |
| Improve | none / truth_priority / anti_sycophancy / verify_then_answer |
| Activation Patching | 100 条样本，24 层逐层 patch |

数据构造中，`full_question` 来自 plain MMLU，`opinion` 来自 opinion-only 数据，因此 plain 条件不包含用户观点干扰；opinion 条件显式加入错误用户观点，用于观察正确答案与用户观点对应选项之间的竞争。

## 4. Locate：关键层定位

### 4.1 行为层结果

`locate_behavior_summary.csv` 显示，在 100 条样本上，三种 prompt 条件的总体行为如下：

| condition | n | accuracy | sycophancy_rate |
|---|---:|---:|---:|
| opinion_only | 100 | 0.32 | 0.23 |
| plain | 100 | 0.35 | 0.24 |
| prefix_and_opinion | 100 | 0.31 | 0.23 |

解释：Qwen2.5-0.5B 在这个 MMLU 子集上的正确率较低，说明小模型本身事实能力有限；同时，加入用户观点后 sycophancy rate 没有明显上升，说明最终离散答案层面并未出现强烈的迎合放大。后续定位分析因此更关注内部层表示中“正确答案 vs 用户观点选项”的概率竞争，而不是只看最终 accuracy。

### 4.2 Logit Lens 层级证据

`locate_layer_summary.csv` 对每层 hidden state 解码，比较 `p_opinion - p_answer`。最高的若干层如下：

| condition | layer | p_answer | p_opinion | opinion_minus_answer |
|---|---:|---:|---:|---:|
| opinion_only | 20 | 0.268818 | 0.299522 | 0.030704 |
| prefix_and_opinion | 20 | 0.273234 | 0.292846 | 0.019612 |
| opinion_only | 3 | 0.248705 | 0.263929 | 0.015224 |
| plain | 3 | 0.244740 | 0.257873 | 0.013133 |
| prefix_and_opinion | 3 | 0.249744 | 0.258935 | 0.009191 |

解释：100 样本下，logit lens 中最清晰的 opinion-over-answer 信号出现在第 20 层；早层第 3 层也有弱信号，但其更像通用选项先验或题目分布影响。第 20 层更适合作为 sycophancy 相关表示的候选定位层。

### 4.3 Activation Patching 因果证据

`activation_patching_summary.csv` 将 opinion prompt 的每层最后 token 表示替换为 plain prompt 表示，并观察 correct-minus-opinion margin 的恢复量。聚合后的最高层为：

| layer | n | base_margin | patched_margin | mean_patch_delta |
|---:|---:|---:|---:|---:|
| 22 | 100 | 0.108589 | 1.364783 | 1.256194 |
| 20 | 100 | 0.108589 | 1.209724 | 1.101135 |
| 21 | 100 | 0.108589 | 1.202563 | 1.093974 |
| 23 | 100 | 0.108589 | 1.065940 | 0.957351 |
| 18 | 100 | 0.108589 | 0.946624 | 0.838035 |

解释：patching 与 logit lens 的证据基本对齐。第 20 层在 logit lens 中最突出，而第 20-23 层在 patching 中给出最大的因果恢复。因此，本项目把 Qwen2.5-0.5B 的后段 residual layers（尤其 20-23 层）视为当前实验中最值得干预的区域。

## 5. Steer：推理阶段向量干预

Steering 使用如下向量算术：

```text
steering_vector = mean(hidden_plain[layer] - hidden_opinion[layer])
```

然后在 opinion prompt 推理时，将 `alpha * steering_vector` 注入指定层最后 token 表示。产物包括：

- `steering_vector.pt`
- `steer_sweep.csv`
- `steer_sweep_summary.csv`

100 样本 sweep 结果：

| layer | alpha | n | accuracy | sycophancy_rate |
|---:|---:|---:|---:|---:|
| 23 | -2.0 | 100 | 0.32 | 0.23 |
| 23 | 0.0 | 100 | 0.32 | 0.23 |
| 23 | 2.0 | 100 | 0.32 | 0.23 |

解释：当前 layer 23、alpha = -2/0/2 的单层 steering 没有改变最终离散答案。这不是失败的运行，而是一个有信息量的负结果：小模型在该数据子集上的输出较稳定，单层线性注入还不足以改变答案。结合 locate 结果，后续更合理的扩展应围绕第 20-23 层做更密的 alpha grid，并比较多层联合注入或 ablation。

## 6. Improve：Prompt-level mitigation

Prompt mitigation 产物包括：

- `improve_prompt_mitigation.csv`
- `improve_prompt_mitigation_summary.csv`

100 样本结果：

| mitigation | n | accuracy | sycophancy_rate |
|---|---:|---:|---:|
| anti_sycophancy | 100 | 0.31 | 0.23 |
| none | 100 | 0.32 | 0.23 |
| truth_priority | 100 | 0.32 | 0.23 |
| verify_then_answer | 100 | 0.31 | 0.23 |

解释：简单提示词缓解没有降低 sycophancy rate，也没有提升 accuracy。这说明对小模型和 MMLU 难题而言，“请优先事实”这类提示不足以修复内部表示竞争。更有希望的改进方向是把 prompt mitigation 与第 20-23 层的定位结果结合，做层级 steering sweep 或更细粒度的 attention/MLP patching。

## 7. 论文复现与拓展

本项目复现的核心思想是：当用户给出错误观点时，模型内部表示会在“事实答案”和“用户观点选项”之间产生竞争；mechanistic interpretability 的价值在于不仅报告最终答错/迎合，还要定位这种竞争在哪些层出现，并测试干预能否改变它。

对应 README 的三段链路：

1. **Locate**：Logit Lens 观察层间趋势，Activation Patching 给出因果恢复证据。
2. **Steer**：用 plain-opinion 表示差构造 steering vector，在推理阶段注入。
3. **Improve**：比较提示词缓解，并指出 prompt-only 方法在本实验中的不足。

相对基础复现，本项目增加了：

- 本地 CPU 可运行的小模型实验路径。
- 100 条 MMLU 样本与 56 个 subject 的结果，而不是极小 smoke test。
- activation patching 的逐层 CSV 产物和聚合表。
- 中文讲解 PPT，并使用用户提供的 `智科院紫.pptx` 模板。

## 8. 个人分析与局限

我的判断是：sycophancy 不应只用最终答案是否等于用户观点来解释。对 Qwen2.5-0.5B 这类小模型，最终答案受事实能力、选项先验、提示格式和解码噪声共同影响；更可靠的分析应看内部表示中正确答案与用户观点选项的竞争轨迹。

本次结果中，logit lens 指向第 20 层，activation patching 指向第 20-23 层。这说明“用户观点相关表示”更像是在后段 residual stream 中被整合进输出决策，而不是只在最后 lm head 才突然出现。

局限：

- 样本量为 100，已经比 smoke test 更可靠，但仍不足以做严格统计显著性结论。
- Steering 只覆盖 layer 23 和三个 alpha，尚未充分利用 patching 定位出的第 20-22 层。
- Patching 目前是 residual layer 粒度，还没有拆到 attention head / MLP output。
- Qwen2.5-0.5B 事实能力有限，accuracy 偏低会影响 sycophancy 行为指标的解释。

后续改进：

- 扩展到 300-500 条样本，并按 subject 分组汇总。
- 对 layer 20-23 做 alpha = -4,-2,-1,0,1,2,4 的 sweep。
- 进一步做 attention output 和 MLP output patching，定位更细模块。
- 使用 Qwen2.5-1.5B 或 Llama 小模型对比模型规模是否改变关键层位置。

## 9. 交付物清单

| 类型 | 路径 |
|---|---|
| 代码 | `outputs/code/` |
| 输入数据 | `outputs/qwen_mmlu_100.csv` |
| 实验结果 | `outputs/qwen_cpu_mmlu100_locate_steer_improve/` |
| 中文报告 | `outputs/report_zh.md` |
| 中文 PPT | `outputs/LLM_sycophancy_locate_steer_improve_presentation.pptx` |
| PPT 模板副本 | `inputs/智科院紫.pptx`, `inputs/template_zky_purple.pptx` |
| PPT 映射/审计 | `outputs/template-frame-map.json`, `outputs/template-audit.txt`, `outputs/deviation-log.txt` |

