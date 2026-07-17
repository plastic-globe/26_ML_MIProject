# LLM Sycophancy Locate-Steer-Improve

本仓库用于复现和扩展大语言模型在 MMLU 事实性选择题中的 sycophancy（迎合错误用户观点）现象。当前版本已经从早期 smoke / 3000 条子集实验，整理为基于 `mmlu_raw` 全量数据的 Locate-Steer-Improve 链路，并补充了 Qwen2.5-0.5B 与 Qwen2.5-1.5B 的全量对照结果、中文报告和中文 PPT。

## References
- 原仓库地址 ： https://github.com/kaustpradalab/LLM-sycophancy
A practical review of mechanistic interpretability for transformer-based language models. Arxiv 2024.
Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in LLMs. Arxiv 2026 

## 当前状态

- 全量数据：`raw_data/mmlu_raw.pkl`，共 14042 条，57 个 subject。
- 全量输入 CSV：`outputs/qwen_mmlu_raw_full.csv`。
- 主实验模型：
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- 运行框架：PyTorch + Hugging Face Transformers + CUDA。
- 远程运行环境：AutoDL / SeetaCloud GPU。
- 代码入口已收敛到 `scripts/`；不再使用重复的 `outputs/code/` 作为运行代码目录。

## 目录结构

```text
D:\26_ML_MIProject
|-- experiments/                         # 原始分模块实验脚本
|   |-- behavioral_analysis/
|   |-- mechanistic_analysis/
|   |-- steering_analysis/
|   `-- improvement_analysis/
|-- lib/                                 # plain / opinion / pov 数据版本
|-- raw_data/                            # mmlu_raw.pkl 等原始数据
|-- scripts/                             # 当前推荐使用的脚本入口
|-- outputs/                             # 全量实验结果、报告、PPT、QA 图
|-- utils/                               # 分析工具类
|-- requirements.txt
`-- README.md
```

## 推荐代码入口

当前新增和整理后的脚本统一放在 `scripts/`：

| 脚本 | 用途 |
|---|---|
| `scripts/build_mmlu_raw_full_csv.py` | 从 `raw_data/mmlu_raw.pkl` 和 `lib/` 构建全量实验 CSV |
| `scripts/run_qwen3000_cpu_suite.py` | Qwen Locate-Steer-Improve 主实验 runner，名称保留历史痕迹，但支持全量数据 |
| `scripts/remote_seeta_job.py` | 上传项目、启动远程实验、下载结果、执行远程命令 |
| `scripts/summarize_full_results.py` | 生成全量结果统计、SVG 图表和中文报告 |
| `scripts/build_full_sycophancy_presentation.mjs` | 生成 0.5B 全量中文 PPT |
| `scripts/build_qwen15_sycophancy_presentation.mjs` | 生成 1.5B 全量中文 PPT |
| `scripts/run_activation_patching_qwen.py` | 独立 activation patching 实验脚本 |

说明：`outputs/code/` 中的重复运行代码没有保留为目标目录入口；如果要复现实验，请使用 `scripts/` 下的版本。

## 环境准备

本地已有 `.venv312` 时可直接使用：

```powershell
cd D:\26_ML_MIProject
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` 当前包含：

```text
torch
transformers
pandas
tqdm
accelerate
vllm
datasets
```

如果需要访问 Hugging Face 模型，请在 `config.py` 中配置 token，或在运行环境中设置 Hugging Face 相关环境变量。远程环境若无法直连 Hugging Face，可使用镜像端点：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 数据准备

当前全量输入已经生成：

```text
outputs/qwen_mmlu_raw_full.csv
```

重新生成命令：

```powershell
cd D:\26_ML_MIProject
.\.venv312\Scripts\python.exe scripts\build_mmlu_raw_full_csv.py
```

构建逻辑：

- `full_question` 来自 `lib/plain/mmlu_plain.pkl`
- `opinion` 来自 `lib/opinion_only/prefix/mmlu_opinion_only.pkl`
- 按 `uid` 与 `raw_data/mmlu_raw.pkl` 合并
- 校验 `answer == opinion` 的行数为 0

## 运行全量 Qwen 实验

### 0.5B 全量实验

```powershell
cd D:\26_ML_MIProject
.\.venv312\Scripts\python.exe scripts\run_qwen3000_cpu_suite.py `
  --model_alias qwen2.5-0.5b `
  --input_file outputs\qwen_mmlu_raw_full.csv `
  --output_dir outputs\qwen_mmlu_raw_full_locate_steer_improve `
  --max_examples 14042 `
  --batch_size 128 `
  --patch_batch_size 32 `
  --train_size 1024 `
  --device cuda `
  --layers 20,21,22,23 `
  --alphas -4,-2,-1,0,1,2,4 `
  --mitigation_modes none,truth_priority,anti_sycophancy,verify_then_answer,counter_opinion_check `
  --stages behavior,logit_lens,steer,patching,improve,elimination
```

### 1.5B 全量实验

```powershell
cd D:\26_ML_MIProject
.\.venv312\Scripts\python.exe scripts\run_qwen3000_cpu_suite.py `
  --model_alias qwen2.5-1.5b `
  --input_file outputs\qwen_mmlu_raw_full.csv `
  --output_dir outputs\qwen15_mmlu_raw_full_locate_steer_improve `
  --max_examples 14042 `
  --batch_size 64 `
  --patch_batch_size 16 `
  --train_size 1024 `
  --device cuda `
  --layers 24,25,26,27 `
  --alphas -4,-2,-1,0,1,2,4 `
  --mitigation_modes none,truth_priority,anti_sycophancy,verify_then_answer,counter_opinion_check `
  --stages behavior,logit_lens,steer,patching,improve,elimination
```

### 远程 AutoDL / SeetaCloud 辅助脚本

`scripts/remote_seeta_job.py` 支持上传、启动、下载、状态检查和远程命令执行。密码不写入仓库，请用环境变量传入：

```powershell
$env:SEETA_PASSWORD="your_password"
.\.venv312\Scripts\python.exe scripts\remote_seeta_job.py status
Remove-Item Env:\SEETA_PASSWORD
```

## 生成报告与 PPT

### 0.5B 中文报告与图表

```powershell
.\.venv312\Scripts\python.exe scripts\summarize_full_results.py `
  --base outputs\qwen_mmlu_raw_full_locate_steer_improve `
  --analysis-dir outputs\full_analysis `
  --report outputs\report_zh_full.md `
  --label Qwen2.5-0.5B-Instruct
```

### 1.5B 中文报告与图表

```powershell
.\.venv312\Scripts\python.exe scripts\summarize_full_results.py `
  --base outputs\qwen15_mmlu_raw_full_locate_steer_improve `
  --analysis-dir outputs\full_analysis_qwen15 `
  --report outputs\report_zh_full_qwen15.md `
  --label Qwen2.5-1.5B-Instruct
```

### 中文 PPT

```powershell
node scripts\build_full_sycophancy_presentation.mjs
node scripts\build_qwen15_sycophancy_presentation.mjs
```

当前 PPT 生成脚本会导出 PPTX、逐页 PNG 预览和 inspect 文件。若当前 Windows 环境中缺少 `pdf2image`，`slides_test.py` 可能无法运行；本仓库已保留 artifact-tool 逐页 PNG 和 contact sheet 作为视觉 QA 依据。

## 当前交付物

### 0.5B 全量结果

- 实验目录：`outputs/qwen_mmlu_raw_full_locate_steer_improve/`
- 中文报告：`outputs/report_zh_full.md`
- 中文 PPT：`outputs/LLM_sycophancy_locate_steer_improve_presentation_full.pptx`
- 图表目录：`outputs/full_analysis/`
- PPT 总览图：`outputs/final-contact-sheet-full.png`

### 1.5B 全量结果

- 实验目录：`outputs/qwen15_mmlu_raw_full_locate_steer_improve/`
- 中文报告：`outputs/report_zh_full_qwen15.md`
- 中文 PPT：`outputs/LLM_sycophancy_locate_steer_improve_presentation_qwen15.pptx`
- 图表目录：`outputs/full_analysis_qwen15/`
- PPT 总览图：`outputs/final-contact-sheet-qwen15.png`

更多文件索引见：

```text
outputs/DELIVERABLES.md
outputs/qa-notes.txt
```

## 全量结果摘要

### 行数校验

| 模型 | Behavior | Logit Lens | Steering | Prompt Improve | Patching |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B | 42126 | 1011024 | 393176 | 70210 | 337008 |
| Qwen2.5-1.5B | 42126 | 1179528 | 393176 | 70210 | 393176 |

### 行为结果

| 模型 | plain acc | plain syc | opinion-only acc | opinion-only syc |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B | 43.9% | 19.1% | 17.8% | 69.2% |
| Qwen2.5-1.5B | 57.5% | 14.7% | 47.1% | 32.5% |

### 主要发现

- 0.5B 在 opinion-only 条件下迎合率升至 69.2%，错误用户观点对输出影响很强。
- 1.5B 的 opinion-only 迎合率降至 32.5%，说明较大模型更稳，但迎合没有消失。
- 0.5B 的 Logit Lens 在 layer 20-22 出现明显 opinion-over-answer 峰值；1.5B 的最大 gap 只有约 0.028，主要在早层。
- 1.5B 的 activation patching 在 layer 26、23、22 对 correct-minus-opinion margin 增强最明显，且 base margin 已为正。
- 简单 mean-vector steering 对 1.5B 的最佳消除率约 0.9%，弱于 prompt mitigation。
- `anti_sycophancy` 是两轮全量实验中最稳定的 prompt-level mitigation。

## QA 与复核

可用以下命令快速复核 1.5B 全量输出：

```powershell
.\.venv312\Scripts\python.exe -c "import pandas as pd, json, pathlib; b=pathlib.Path('outputs/qwen15_mmlu_raw_full_locate_steer_improve'); print(json.load(open(b/'run_config.json', encoding='utf-8'))['num_examples']); expected={'locate_behavior.csv':42126,'locate_layer_logit_lens.csv':1179528,'steer_sweep.csv':393176,'improve_prompt_mitigation.csv':70210,'activation_patching_summary.csv':393176}; [print(k, len(pd.read_csv(b/k)), v) for k,v in expected.items()]"
```

也可查看：

```text
outputs/qa-notes.txt
```

## 维护说明

- 新增实验运行代码请优先放入 `scripts/`。
- 生成产物放入 `outputs/`，不要在 `outputs/` 下再放运行代码副本。
- 原始分模块脚本仍保留在 `experiments/`，用于追溯和扩展。
- 大文件结果已经保存在 `outputs/`，提交或传输前请根据需要确认 `.gitignore` 策略。
