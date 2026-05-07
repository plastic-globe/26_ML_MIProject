# 26_ML_MIProject

Paper reproduction project for mechanistic interpretability, centered on:

- locating layers and modules responsible for sycophancy
- steering model behavior during inference
- reproducing the paper `When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models`

## Project Layout

- `scripts/`: entry-point scripts for behavior evaluation and layer-wise analysis
- `src/mi_repro/`: shared utilities for prompts, data loading, metrics, and logit lens decoding
- `data/`: local MMLU-style datasets
- `results/`: exported metrics and traces
- `notebooks/`: ad hoc analysis and plotting
- `configs/`: future experiment configs

## First Scripts

- `scripts/eval_behavior.py`
  Runs `plain`, `opinion_only`, `first_person`, and `third_person` prompt conditions and reports:
  - accuracy
  - sycophancy rate
  - independent error rate

- `scripts/logit_lens.py`
  Extracts hidden states for each layer, decodes answer-option logits with a logit-lens style projection, and exports:
  - per-layer logits for `A/B/C/D`
  - per-layer decision scores for the correct and user-suggested wrong answer

## Expected Dataset Format

Place a `.csv` or `.jsonl` file in `data/`.

CSV columns:

- `question`
- `A`
- `B`
- `C`
- `D`
- `answer`
- `subject`

## Example Commands

```powershell
$env:PYTHONPATH="src"
python scripts/eval_behavior.py --model meta-llama/Llama-3.1-8B-Instruct --data-path data/mmlu_sample.csv --limit 100
python scripts/logit_lens.py --model meta-llama/Llama-3.1-8B-Instruct --data-path data/mmlu_sample.csv --limit 50
```

## Next Recommended Steps

- add `activation_patching.py`
- add steering-vector extraction and intervention
- add plotting utilities for the exported CSV files
