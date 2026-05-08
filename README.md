# 26_ML_MIProject

## 项目简介
- NJU智能科学与技术学院，机器学习导论课程复现论文项目，原论文仓库：github.com/kaustpradalab/LLM-sycophancy
- 基于 logit_lens 和 activation_patches 方法，分析模型出现“阿谀奉承”现象背后的机制。


## 项目结构
```
.
├── scripts/
│   ├── convert_mmlu.py          # Data preparation: export cais/mmlu to local CSV/JSONL
│   ├── eval_behavior.py         # Section: behavioral reproduction
│   └── logit_lens.py            # Section: mechanistic analysis
├── src/
│   └── mi_repro/
│       ├── prompts.py           # Prefix and prompt generation
│       ├── data.py              # Dataset parsing and example construction
│       ├── modeling.py          # Hugging Face model interface
│       ├── metrics.py           # Accuracy / sycophancy / independent error
│       └── logit_lens_utils.py  # Logit-lens decoding and layer-wise scoring
├── data/                        # Local converted MMLU files
├── results/                     # Exported experimental outputs
├── notebooks/                   # Visualization and result inspection
├── configs/                     # Future config files
├── requirements.txt
└── README.md
```

## 本地使用