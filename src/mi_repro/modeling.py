"""Model loading and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device


def load_model(model_name_or_path: str, *, device: str = "auto", dtype: str = "auto") -> LoadedModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_dtype(dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    resolved_device = _infer_primary_device(model)
    return LoadedModel(tokenizer=tokenizer, model=model, device=resolved_device)


@torch.inference_mode()
def generate_single_token_choice(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 3,
) -> str:
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.model.device)
    outputs = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        pad_token_id=loaded.tokenizer.pad_token_id,
    )
    completion_ids = outputs[0, inputs["input_ids"].shape[1] :]
    text = loaded.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text


def answer_choice_logits(
    loaded: LoadedModel,
    prompt: str,
    *,
    output_hidden_states: bool = False,
):
    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.model.device)
    return loaded.model(**inputs, output_hidden_states=output_hidden_states)


def token_id_for_single_letter(loaded: LoadedModel, letter: str) -> int:
    ids = loaded.tokenizer.encode(letter, add_special_tokens=False)
    if len(ids) != 1:
        prefixed = loaded.tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(prefixed) != 1:
            raise ValueError(f"Could not resolve {letter!r} to a single token")
        return prefixed[0]
    return ids[0]


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def _infer_primary_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

