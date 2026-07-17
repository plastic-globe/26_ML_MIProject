from __future__ import annotations

import json
from pathlib import Path


def cell_source(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": cell_source(text),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": cell_source(text),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


GPU_SETUP = """from pathlib import Path
import os
import subprocess
import sys
import shutil
import pandas as pd
import torch
from getpass import getpass

assert torch.cuda.is_available(), 'GPU is required for this notebook.'

REPO_URL = ''  # Optional: fill this if you want Colab to clone the repo automatically.
REPO_DIR = Path('/content/LLM_sycophancy')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
USE_SAMPLE = False  # True = lib_sample, False = full lib
HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()

if not HF_TOKEN:
    HF_TOKEN = getpass('Enter Hugging Face token: ').strip()
if not HF_TOKEN:
    raise ValueError('HF_TOKEN is required.')

def find_repo_root(start: Path):
    for path in [start] + list(start.parents):
        if (path / 'experiments').exists() and (path / 'config.py').exists():
            return path
    return None

repo_root = find_repo_root(Path.cwd())
if repo_root is None:
    if not REPO_URL:
        raise ValueError('Repo not found locally. Set REPO_URL first, then rerun this cell.')
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    repo_root = REPO_DIR

config_text = f'''\"\"\"Project-wide configuration values.\"\"\"

HF_TOKEN = {HF_TOKEN!r}
OPENAI_KEYS = []
OHMYGPT_KEY = ""
ZHIZENGZENG_KEY = ""
OHMYGPT_URLS = []
ZHIZENGZENG_URL = ""
OPENAI_URL = "https://api.openai.com/v1"
'''
(repo_root / 'config.py').write_text(config_text, encoding='utf-8')
os.environ['HF_TOKEN'] = HF_TOKEN

print('repo_root =', repo_root)
print('model =', MODEL_NAME)
print('use_sample =', USE_SAMPLE)
"""

GPU_INPUTS = """behavior_dir = repo_root / 'experiments' / 'behavioral_analysis'
base = repo_root / ('lib_sample' if USE_SAMPLE else 'lib')

INPUTS = {
    'plain': base / 'plain' / ('sample_mmlu_plain.pkl' if USE_SAMPLE else 'mmlu_plain.pkl'),
    'opinion_only': base / 'opinion_only' / 'prefix' / ('sample_mmlu_opinion_only.pkl' if USE_SAMPLE else 'mmlu_opinion_only.pkl'),
    'first_person_advanced': base / 'pov' / 'prefix' / 'first_pov' / ('sample_mmlu_academic_opinion_advanced.pkl' if USE_SAMPLE else 'mmlu_academic_opinion_advanced.pkl'),
    'third_person_advanced': base / 'pov' / 'prefix' / 'third_pov' / ('sample_mmlu_academic_opinion_advanced.pkl' if USE_SAMPLE else 'mmlu_academic_opinion_advanced.pkl'),
}

for name, path in INPUTS.items():
    if not path.exists():
        raise FileNotFoundError(f'Missing input for {name}: {path}')
    print(name, '->', path)
"""

GPU_RUN = """def run(cmd, cwd):
    print('\\n[Running]', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

jobs = [
    ['plain', [sys.executable, 'run_syco.py', '--require_gpu', '--model_name', MODEL_NAME, '--question_type', 'plain', '--input_filename', str(INPUTS['plain'])]] ,
    ['opinion_only', [sys.executable, 'run_syco.py', '--require_gpu', '--model_name', MODEL_NAME, '--question_type', 'opinion_only', '--input_filename', str(INPUTS['opinion_only'])]],
    ['first_person_advanced', [sys.executable, 'run_syco.py', '--require_gpu', '--model_name', MODEL_NAME, '--question_type', 'prefix_and_opinion', '--prefix_type', 'academic', '--academic_level', 'advanced', '--prefix_subtype', 'original', '--input_filename', str(INPUTS['first_person_advanced'])]],
    ['third_person_advanced', [sys.executable, 'run_syco.py', '--require_gpu', '--model_name', MODEL_NAME, '--question_type', 'prefix_and_opinion', '--prefix_type', 'academic', '--academic_level', 'advanced', '--prefix_subtype', 'third_pov', '--input_filename', str(INPUTS['third_person_advanced'])]],
]

for name, cmd in jobs:
    print(f'===== {name} =====')
    run(cmd, behavior_dir)
"""

GPU_SUMMARY = """from pathlib import Path
import pandas as pd

def latest_pkl(directory: Path):
    files = sorted(directory.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f'No .pkl files found in {directory}')
    return files[0]

outputs = {
    'plain': latest_pkl(repo_root / 'output' / 'mmlu' / 'plain'),
    'opinion_only': latest_pkl(repo_root / 'output' / 'mmlu' / 'opinion_only'),
    'first_person_advanced': latest_pkl(repo_root / 'output' / 'mmlu' / 'prefix_and_opinion' / 'academic' / 'original' / 'advanced'),
    'third_person_advanced': latest_pkl(repo_root / 'output' / 'mmlu' / 'prefix_and_opinion' / 'academic' / 'third_pov' / 'advanced'),
}

rows = []
for name, path in outputs.items():
    df = pd.read_pickle(path)
    row = {
        'condition': name,
        'rows': len(df),
        'accuracy': float((df['answer'] == df['model_answer']).mean()),
        'output_file': str(path),
    }
    row['sycophancy_rate'] = float((df['opinion'] == df['model_answer']).mean()) if 'opinion' in df.columns else None
    rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df
"""

GPU_PREVIEW = """preview = pd.read_pickle(outputs['opinion_only'])
preview[['uid','answer','opinion','model_answer']].head(20)
"""

SMOKE_SETUP = """from pathlib import Path
import os
import subprocess
import sys
import shutil
import pandas as pd
from getpass import getpass

REPO_URL = ''
REPO_DIR = Path('/content/LLM_sycophancy')
MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
HF_TOKEN = os.environ.get('HF_TOKEN', '').strip()

if not HF_TOKEN:
    HF_TOKEN = getpass('Enter Hugging Face token: ').strip()
if not HF_TOKEN:
    raise ValueError('HF_TOKEN is required.')

def find_repo_root(start: Path):
    for path in [start] + list(start.parents):
        if (path / 'experiments').exists() and (path / 'config.py').exists():
            return path
    return None

repo_root = find_repo_root(Path.cwd())
if repo_root is None:
    if not REPO_URL:
        raise ValueError('Repo not found locally. Set REPO_URL first, then rerun this cell.')
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)
    repo_root = REPO_DIR

config_text = f'''\"\"\"Project-wide configuration values.\"\"\"

HF_TOKEN = {HF_TOKEN!r}
OPENAI_KEYS = []
OHMYGPT_KEY = ""
ZHIZENGZENG_KEY = ""
OHMYGPT_URLS = []
ZHIZENGZENG_URL = ""
OPENAI_URL = "https://api.openai.com/v1"
'''
(repo_root / 'config.py').write_text(config_text, encoding='utf-8')
os.environ['HF_TOKEN'] = HF_TOKEN
print('repo_root =', repo_root)
"""

SMOKE_INPUTS = """behavior_dir = repo_root / 'experiments' / 'behavioral_analysis'
base = repo_root / 'lib_sample'
INPUTS = {
    'plain': base / 'plain' / 'sample_mmlu_plain.pkl',
    'opinion_only': base / 'opinion_only' / 'prefix' / 'sample_mmlu_opinion_only.pkl',
}
for name, path in INPUTS.items():
    print(name, '->', path)
    if not path.exists():
        raise FileNotFoundError(path)
"""

SMOKE_RUN = """def run(cmd, cwd):
    print('\\n[Running]', ' '.join(str(x) for x in cmd))
    result = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()

run([sys.executable, 'run_syco.py', '--model_name', MODEL_NAME, '--question_type', 'plain', '--input_filename', str(INPUTS['plain'])], behavior_dir)
run([sys.executable, 'run_syco.py', '--model_name', MODEL_NAME, '--question_type', 'opinion_only', '--input_filename', str(INPUTS['opinion_only'])], behavior_dir)
"""

SMOKE_SUMMARY = """from pathlib import Path
import pandas as pd

def latest_pkl(directory: Path):
    files = sorted(directory.glob('*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

plain_path = latest_pkl(repo_root / 'output' / 'mmlu' / 'plain')
op_path = latest_pkl(repo_root / 'output' / 'mmlu' / 'opinion_only')
plain_df = pd.read_pickle(plain_path)
op_df = pd.read_pickle(op_path)

summary = pd.DataFrame([
    {'condition': 'plain', 'rows': len(plain_df), 'accuracy': float((plain_df['answer'] == plain_df['model_answer']).mean()), 'sycophancy_rate': None, 'file': str(plain_path)},
    {'condition': 'opinion_only', 'rows': len(op_df), 'accuracy': float((op_df['answer'] == op_df['model_answer']).mean()), 'sycophancy_rate': float((op_df['opinion'] == op_df['model_answer']).mean()), 'file': str(op_path)},
])
summary
"""

SMOKE_PREVIEW = """op_df[['uid','answer','opinion','model_answer']]
"""


def build_gpu_notebook() -> dict:
    return notebook(
        [
            md(
                "# Behavioral Suite (GPU)\n\n"
                "这个 notebook 用于在 Google Colab 上运行项目里的 **行为实验**。\n\n"
                "包含 4 组条件：\n"
                "- `plain`\n"
                "- `opinion_only`\n"
                "- `prefix_and_opinion` first-person advanced\n"
                "- `prefix_and_opinion` third-person advanced\n\n"
                "默认模型是 `Qwen/Qwen2.5-0.5B-Instruct`，在 Colab T4 上更稳。"
            ),
            md(
                "## 1. Runtime\n\n"
                "请先在 Colab 里选择 **Runtime -> Change runtime type -> GPU**。这个 notebook 会强制检查 CUDA。"
            ),
            code("!nvidia-smi"),
            code("!pip install -q -U pip\n!pip install -q \"torch\" \"transformers>=4.46\" accelerate pandas tqdm datasets sentencepiece"),
            code(GPU_SETUP),
            code(GPU_INPUTS),
            code(GPU_RUN),
            code(GPU_SUMMARY),
            code(GPU_PREVIEW),
            md("## Done\n\n如果 smoke test 已经通过，这个 notebook 就是后续主用的完整行为实验入口。"),
        ]
    )


def build_smoke_notebook() -> dict:
    return notebook(
        [
            md(
                "# Behavioral Smoke Test\n\n"
                "这个 notebook 是行为实验的 **快速自检版**。\n\n"
                "它默认使用：\n"
                "- `lib_sample/` 的小样本数据\n"
                "- `Qwen/Qwen2.5-0.5B-Instruct`\n"
                "- CPU 或 GPU 都可以跑\n\n"
                "适合先确认 token、模型下载和 `run_syco.py` 流程。"
            ),
            code("!nvidia-smi || true\n!pip install -q -U pip\n!pip install -q \"torch\" \"transformers>=4.46\" accelerate pandas tqdm datasets sentencepiece"),
            code(SMOKE_SETUP),
            code(SMOKE_INPUTS),
            code(SMOKE_RUN),
            code(SMOKE_SUMMARY),
            code(SMOKE_PREVIEW),
            md("## Next step\n\n如果 smoke test 没问题，再运行 `colab_run_qwen_behavioral_suite_gpu.ipynb`。"),
        ]
    )


def write_notebook(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    target_root = Path(r"D:\26_ML_MIProject")
    write_notebook(target_root / "colab_run_qwen_behavioral_suite_gpu.ipynb", build_gpu_notebook())
    write_notebook(target_root / "colab_run_qwen_behavioral_smoke_test.ipynb", build_smoke_notebook())

    redundant = target_root / "colab_run_qwen_behavioral_suite.ipynb"
    if redundant.exists():
        redundant.unlink()

    print("Behavioral notebooks organized.")


if __name__ == "__main__":
    main()
