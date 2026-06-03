<div align="center">

## LLM Sycophancy 复现与改进项目

基于 [kaustpradalab/LLM-sycophancy](https://github.com/kaustpradalab/LLM-sycophancy.git) 的复现、整理与扩展实现

</div>

<p align="center">
    <img src="img/overview.png" alt="overview" width="70%">
</p>

## 项目简介

本项目围绕大语言模型中的 **sycophancy（迎合）** 现象展开，目标是在原始工作的基础上完成更完整的 `locate -> steer -> improve` 研究链路：

- `locate`：复现并分析模型在何种提示条件下出现迎合，以及这种行为在层级表示中的出现位置
- `steer`：加入基于激活向量的 steering 方法，尝试把模型从“迎合用户错误观点”拉回到“恢复原本的正确判断”
- `improve`：加入 prompt-level mitigation 和 steering 参数搜索，评估可操作的改进策略

和上游仓库相比，这个版本不只保留了原有的行为实验与机理分析，还补充了仓库里原先缺失的 `steer` 与 `improve` 部分，并对数据构建流程做了统一整理。

## 当前仓库覆盖内容

### 1. Locate：定位迎合行为的触发条件与内部层位

- `experiments/behavioral_analysis/run_syco.py`
  - 运行 plain、opinion_only、prefix_and_opinion 三类提示下的推理
  - 用于比较基线准确率、迎合率，以及不同 persona / POV 条件下的行为差异
- `experiments/mechanistic_analysis/run_syco_logit_cot.py`
  - 提取逐层 logits，并支持 `logit_only` / `logit_and_cot`
  - 用于分析模型在不同层上的答案偏移情况
- `utils/SycophancyAnalysis.py`
  - 面向已有层级输出结果做后处理和可视化
- `utils/EarlyDecodingAnalysis.py`
  - 用于分析早期层解码、预测稳定层、plain vs opinion 的层级差异

### 2. Steer：激活向量干预

- `experiments/steering_analysis/run_activation_steering.py`
  - 从 `plain` 与 `opinion_only` 对齐样本中学习 steering vector
  - 在指定 decoder layer 注入激活偏移
  - 输出 baseline / steered 的正确率与迎合率对比
- `utils/SteeringAnalysis.py`
  - 汇总 `output_steering/` 下的 steering 结果并排序

### 3. Improve：缓解与优化

- `experiments/improvement_analysis/run_prompt_mitigation.py`
  - 评估不同 prompt-level mitigation 提示词
  - 当前内置模式包括 `truth_priority`、`anti_sycophancy`、`verify_then_answer`
- `experiments/improvement_analysis/run_steering_sweep.py`
  - 批量搜索不同 `layer_index / alpha / train_size` 组合
  - 自动调用 steering 脚本并生成汇总表
- `utils/ImprovementAnalysis.py`
  - 汇总 prompt mitigation 和 steering sweep 的最优配置

### 4. 数据构建

- `experiments/data_generation/build_lib.py`
  - 这是当前仓库里最完整、最推荐的数据构建脚本
  - 支持从本地文件或 Hugging Face 数据集生成统一的 `lib/` 目录结构
- `experiments/data_generation/generate_prefixes.py`
- `experiments/data_generation/full_question_builder.py`
- `experiments/data_generation/apply_prefixes.py`
  - 这些脚本更偏向早期/辅助数据处理流程，主实验优先使用 `build_lib.py`

## 仓库结构

```text
.
|-- experiments/
|   |-- behavioral_analysis/
|   |   |-- run_syco.py
|   |   `-- run_syco.slurm
|   |-- mechanistic_analysis/
|   |   |-- run_syco_logit_cot.py
|   |   `-- run_syco_logit_cot.slurm
|   |-- steering_analysis/
|   |   `-- run_activation_steering.py
|   |-- improvement_analysis/
|   |   |-- run_prompt_mitigation.py
|   |   `-- run_steering_sweep.py
|   `-- data_generation/
|       |-- build_lib.py
|       |-- generate_prefixes.py
|       |-- full_question_builder.py
|       `-- apply_prefixes.py
|-- utils/
|   |-- SycophancyAnalysis.py
|   |-- EarlyDecodingAnalysis.py
|   |-- SteeringAnalysis.py
|   `-- ImprovementAnalysis.py
|-- lib/
|   |-- plain/
|   |-- opinion_only/
|   `-- pov/
|-- raw_data/
|   |-- sample_mmlu.jsonl
|   `-- mmlu_raw.pkl
|-- img/
|-- DATA_STRUCTURE.md
|-- requirements.txt
`-- README.md
```

## 环境准备

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

- `torch`
- `transformers`
- `pandas`
- `tqdm`
- `accelerate`
- `vllm`
- `datasets`

### 2. 配置 Hugging Face 访问令牌

当前仓库中的相关实验脚本已经统一通过项目根目录下的 `config.py` 读取 `HF_TOKEN`。

请在根目录的 `config.py` 中填写：

```python
# config.py
HF_TOKEN = "your_hf_token"
```

## 数据准备

仓库中已经包含一份处理后的数据目录：

- `raw_data/mmlu_raw.pkl`
- `lib/plain/mmlu_plain.pkl`
- `lib/opinion_only/...`
- `lib/pov/...`

如果你希望从头重建数据，可以使用：

```bash
python experiments/data_generation/build_lib.py \
  --input-file raw_data/sample_mmlu.jsonl
```

或者从 Hugging Face 拉取：

```bash
python experiments/data_generation/build_lib.py \
  --hf-dataset cais/mmlu \
  --hf-config all \
  --hf-split test
```

`build_lib.py` 会生成：

- `raw_data/mmlu_raw.pkl`
- `lib/plain/mmlu_plain.pkl`
- `lib/opinion_only/prefix/mmlu_opinion_only.pkl`
- `lib/opinion_only/suffix/mmlu_opinion_only.pkl`
- `lib/pov/prefix/...`
- `lib/pov/suffix/...`

更详细的数据组织说明见 [DATA_STRUCTURE.md](DATA_STRUCTURE.md)。

## 运行方式

### 一、Locate：行为复现与层级定位

### 1. Plain 基线

```bash
cd experiments/behavioral_analysis

python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type plain \
  --input_filename ../../lib/plain/mmlu_plain.pkl
```

### 2. Opinion-only 条件

```bash
cd experiments/behavioral_analysis

python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type opinion_only \
  --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl
```

### 3. Prefix + Opinion 条件

```bash
cd experiments/behavioral_analysis

python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype original \
  --input_filename ../../lib/pov/prefix/first_pov/mmlu_academic_opinion_advanced.pkl
```

### 4. 第一人称 vs 第三人称

第一人称：

```bash
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype original \
  --input_filename ../../lib/pov/prefix/first_pov/mmlu_academic_opinion_advanced.pkl
```

第三人称：

```bash
python run_syco.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type prefix_and_opinion \
  --prefix_type academic \
  --academic_level advanced \
  --prefix_subtype third_pov \
  --input_filename ../../lib/pov/prefix/third_pov/mmlu_academic_opinion_advanced.pkl
```

行为实验输出默认写入：

- `output/{dataset}/{question_type}/...`

### 5. 逐层 logit / CoT 分析

```bash
cd experiments/mechanistic_analysis

python run_syco_logit_cot.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --question_type opinion_only \
  --inference_mode logit_only \
  --inference_layer all \
  --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl
```

常用参数：

- `--inference_mode`: `logit_only` 或 `logit_and_cot`
- `--inference_layer`: `all`、`odd`、`even`、`last`

机理分析输出默认写入：

- `output_inference/{dataset}/{question_type}/...`

如果你在集群环境中运行，也可以直接使用：

- `experiments/behavioral_analysis/run_syco.slurm`
- `experiments/mechanistic_analysis/run_syco_logit_cot.slurm`

### 二、Steer：激活 steering

```bash
cd experiments/steering_analysis

python run_activation_steering.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --plain_filename ../../lib/plain/mmlu_plain.pkl \
  --opinion_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl \
  --layer_index 18 \
  --alpha 4.0 \
  --train_size 512 \
  --eval_size 1000
```

关键参数：

- `--layer_index`：施加 steering 的层
- `--alpha`：干预强度
- `--train_size`：用于学习 steering vector 的对齐样本数
- `--vector_direction`：`restore_plain` 或 `amplify_opinion`

输出默认写入：

- `output_steering/mmlu/activation_steering/*.pkl`
- `output_steering/mmlu/activation_steering/*.json`

汇总结果：

```bash
python utils/SteeringAnalysis.py
```

### 三、Improve：缓解与参数搜索

### 1. Prompt-level mitigation

```bash
cd experiments/improvement_analysis

python run_prompt_mitigation.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --input_filename ../../lib/opinion_only/prefix/mmlu_opinion_only.pkl \
  --mitigation_mode truth_priority \
  --eval_size 1000
```

可选 `--mitigation_mode`：

- `none`
- `truth_priority`
- `anti_sycophancy`
- `verify_then_answer`

### 2. Steering 参数搜索

```bash
cd experiments/improvement_analysis

python run_steering_sweep.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --layer_indices 16,18,20 \
  --alphas 1.0,2.0,4.0 \
  --train_sizes 128,512 \
  --eval_size 500
```

如果需要指定解释器，可以先设置：

```powershell
$env:PYTHON_EXECUTABLE="python"
```

Improve 输出默认写入：

- `output_improvement/mmlu/prompt_mitigation/`
- `output_improvement/mmlu/steering_sweep/`

汇总结果：

```bash
python utils/ImprovementAnalysis.py
```

## 输出目录说明

不同阶段的默认输出目录如下：

- `output/`：行为实验结果
- `output_inference/`：逐层 logit / CoT 推理结果
- `output_steering/`：激活 steering 结果
- `output_improvement/`：缓解实验与参数搜索结果

## 数据文件说明

当前 `lib/` 目录已经包含以下几类实验输入：

- `lib/plain/mmlu_plain.pkl`
  - 不带用户观点的基线问题
- `lib/opinion_only/prefix/mmlu_opinion_only.pkl`
  - 用户错误观点放在问题前面
- `lib/opinion_only/suffix/mmlu_opinion_only.pkl`
  - 用户错误观点放在问题后面
- `lib/pov/prefix/first_pov/*.pkl`
  - 第一人称 + 专业水平 persona
- `lib/pov/prefix/third_pov/*.pkl`
  - 第三人称 + 专业水平 persona

其中 `pov` 文件名中的 `advanced / intermediate / beginner` 对应不同专业水平设定。

### 3. 辅助分析脚本

`utils/SycophancyAnalysis.py` 与 `utils/EarlyDecodingAnalysis.py` 依赖已有实验输出文件做后处理；在首次运行前，请先确保对应输出目录中已经有 `.pkl` 结果。

## 参考项目

- 原始仓库：[kaustpradalab/LLM-sycophancy](https://github.com/kaustpradalab/LLM-sycophancy.git)
- 相关论文：*When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models*

## 致谢

感谢原始仓库与论文作者提供的研究问题、实验框架与分析视角。本项目在此基础上完成了复现、补全与面向 `locate / steer / improve` 的扩展实现。
