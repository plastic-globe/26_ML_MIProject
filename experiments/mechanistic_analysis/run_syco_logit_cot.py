import argparse
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


LOG_PATH = PROJECT_ROOT / "inference_mechanistic.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run mechanistic logit / CoT inference on pre-constructed question sets."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name for inference",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mmlu",
        choices=["mmlu"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--prefix_type",
        type=str,
        default="",
        choices=["", "academic", "behavior"],
        help="Type of prefix used.",
    )
    parser.add_argument(
        "--academic_level",
        type=str,
        default="",
        choices=["", "beginner", "intermediate", "advanced"],
        help="Academic level for academic prefix.",
    )
    parser.add_argument(
        "--prefix_subtype",
        type=str,
        default="original",
        choices=["", "original", "mixing_subject", "first_pov", "third_pov"],
        help="Subtype of prefix.",
    )
    parser.add_argument(
        "--question_type",
        type=str,
        default="plain",
        choices=["prefix_and_opinion", "opinion_only", "plain"],
        help="Type of the questions.",
    )
    parser.add_argument(
        "--input_filename",
        type=str,
        default="../../lib/plain/mmlu_plain.pkl",
        help="Input .pkl file with pre-constructed questions",
    )
    parser.add_argument(
        "--full_question_column",
        type=str,
        default="full_question",
        help="Column containing the full question text.",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="logit_and_cot",
        choices=["logit_only", "logit_and_cot"],
        help="Whether to compute only logits or also generate CoT.",
    )
    parser.add_argument(
        "--inference_layer",
        type=str,
        default="last",
        choices=["all", "odd", "even", "last"],
        help="Layers used for logit extraction.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries for invalid answers.",
    )
    parser.add_argument(
        "--require_gpu",
        action="store_true",
        help="Fail immediately if CUDA is not available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Omit for full runs.",
    )
    return parser.parse_args()


def is_valid_answer(answer):
    return isinstance(answer, str) and len(answer) == 1 and answer in "ABCD"


def get_layer_indices(total_layers, inference_layer):
    if inference_layer == "all":
        return list(range(total_layers))
    if inference_layer == "odd":
        return [i for i in range(total_layers) if i % 2 == 1 or i == total_layers - 1]
    if inference_layer == "even":
        return [i for i in range(total_layers) if i % 2 == 0 or i == total_layers - 1]
    if inference_layer == "last":
        return [total_layers - 1]
    raise ValueError(f"Invalid inference_layer: {inference_layer}")


def load_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name, hf_token):
    common_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "token": hf_token,
        "trust_remote_code": False,
    }

    if torch.cuda.is_available():
        strategies = [
            {"torch_dtype": "auto", "device_map": "auto"},
            {"torch_dtype": torch.float16},
            {},
        ]
    else:
        strategies = [
            {"torch_dtype": torch.float32},
            {},
        ]

    errors = []
    for strategy in strategies:
        try:
            logging.info("Trying model load strategy: %s", strategy)
            model = AutoModelForCausalLM.from_pretrained(
                **common_kwargs,
                **strategy,
            )
            if "device_map" not in strategy:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            model.eval()
            return model
        except Exception as exc:
            errors.append((strategy, str(exc), traceback.format_exc()))
            logging.error(
                "Model load strategy failed: %s\n%s",
                strategy,
                traceback.format_exc(),
            )

    messages = ["All model loading strategies failed:"]
    for strategy, message, stack in errors:
        messages.append(f"Strategy {strategy}: {message}")
        messages.append(stack)
    raise RuntimeError("\n".join(messages))


def build_prompt(question, inference_mode):
    if inference_mode == "logit_and_cot":
        return (
            f"Question: ||{question}||\n"
            "Think step by step before answering. "
            "At the end, output your final answer in the form {A}, {B}, {C}, or {D}."
        )
    return (
        f"Question: ||{question}||\n"
        "Respond with exactly one uppercase letter (A, B, C, or D) and nothing else.\n"
        "Answer:"
    )


def get_model_input_device(model, fallback_device):
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, str) and first_device not in {"cpu", "disk"}:
            return torch.device(first_device)
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback_device


def parse_cot_answer(raw_output):
    if not raw_output:
        return ""

    stripped = raw_output.strip()
    for marker in ["{A}", "{B}", "{C}", "{D}"]:
        if marker in stripped:
            return marker[1]

    for char in reversed(stripped):
        if char in "ABCD":
            return char
    return ""


def get_answer_token_ids(tokenizer):
    token_ids = {}
    for letter in "ABCD":
        variants = [
            tokenizer.encode(letter, add_special_tokens=False),
            tokenizer.encode(f" {letter}", add_special_tokens=False),
        ]
        valid_ids = [encoded[0] for encoded in variants if len(encoded) == 1]
        if not valid_ids:
            raise ValueError(f"Could not find single-token encoding for answer option {letter}.")
        token_ids[letter] = sorted(set(valid_ids))
    return token_ids


def compute_layer_logits(model, hidden_states, answer_token_ids, layer_indices):
    layer_logits = {}
    for layer_idx in layer_indices:
        hidden_state = hidden_states[layer_idx + 1][:, -1, :]
        projected = model.lm_head(hidden_state)
        answer_scores = {}
        for letter, token_id_candidates in answer_token_ids.items():
            values = [projected[0, token_id].item() for token_id in token_id_candidates]
            answer_scores[letter] = max(values)
        layer_logits[f"layer_{layer_idx}"] = answer_scores
    return layer_logits


def select_answer_from_last_layer(layer_logits):
    if not layer_logits:
        return "Error"

    last_layer_name = sorted(
        layer_logits.keys(),
        key=lambda name: int(name.split("_")[1]),
    )[-1]
    logits_tensor = torch.tensor([layer_logits[last_layer_name][letter] for letter in "ABCD"])
    probabilities = torch.softmax(logits_tensor, dim=0)
    return "ABCD"[int(torch.argmax(probabilities).item())]


def process_question(
    question,
    tokenizer,
    model,
    model_input_device,
    inference_mode,
    inference_layer,
    question_index,
    answer_token_ids,
):
    try:
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Invalid question at index {question_index}: expected non-empty string, got {question!r}"
            )

        prompt = build_prompt(question, inference_mode)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {key: value.to(model_input_device) for key, value in inputs.items()}

        raw_output = ""
        if inference_mode == "logit_and_cot":
            generated = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_tokens = generated[0][inputs["input_ids"].shape[-1]:]
            raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        total_layers = len(hidden_states) - 1
        layer_indices = get_layer_indices(total_layers, inference_layer)
        layer_logits = compute_layer_logits(model, hidden_states, answer_token_ids, layer_indices)

        if inference_mode == "logit_and_cot":
            answer = parse_cot_answer(raw_output)
            if not is_valid_answer(answer):
                answer = select_answer_from_last_layer(layer_logits)
        else:
            answer = select_answer_from_last_layer(layer_logits)

        if not is_valid_answer(answer):
            logging.warning(
                "Invalid answer at index %s. Raw output: %r",
                question_index,
                raw_output,
            )
            return "Error", layer_logits, raw_output

        return answer, layer_logits, raw_output
    except Exception as exc:
        logging.error(
            "Error processing question at index %s: %s\n%s",
            question_index,
            exc,
            traceback.format_exc(),
        )
        return "Error", {}, "Error in processing"


def build_output_dir(dataset, question_type, prefix_type, prefix_subtype, academic_level):
    parts = [PROJECT_ROOT / "output_inference" / dataset]
    if question_type:
        parts.append(Path(question_type))
    if prefix_type:
        parts.append(Path(prefix_type))
        if prefix_subtype:
            parts.append(Path(prefix_subtype))
        if prefix_type == "academic" and academic_level:
            parts.append(Path(academic_level))

    output_dir = parts[0]
    for part in parts[1:]:
        output_dir = output_dir / part
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    args = parse_args()

    if args.academic_level and args.prefix_type != "academic":
        raise ValueError("The --academic_level argument is only applicable when prefix_type='academic'.")

    if args.question_type == "prefix_and_opinion" and not args.prefix_type:
        raise ValueError(
            "For 'prefix_and_opinion' question_type, a prefix_type must be specified."
        )

    hf_token = config.HF_TOKEN
    if not hf_token:
        raise ValueError("HF_TOKEN is not set in config.py.")

    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is required but CUDA is not available. In Colab, set Runtime > Change runtime type > GPU."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logging.info("Using device: %s", device)

    input_path = Path(args.input_filename)
    if not input_path.is_absolute():
        input_path = (Path.cwd() / input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        print("Loading tokenizer...")
        tokenizer = load_tokenizer(args.model_name, hf_token)

        print("Loading model...")
        model = load_model(args.model_name, hf_token)
        model_input_device = get_model_input_device(model, device)
        answer_token_ids = get_answer_token_ids(tokenizer)

        print(f"Loading DataFrame from {input_path}...")
        df = pd.read_pickle(input_path)
        if args.limit is not None:
            df = df.head(args.limit).copy()
            print(f"Row limit enabled: {len(df)} rows")

        print(f"Loaded DataFrame with {len(df)} entries.")
        logging.info("Loaded DataFrame with %s entries from %s", len(df), input_path)

        if args.full_question_column not in df.columns:
            raise ValueError(
                f"Input DataFrame must contain a '{args.full_question_column}' column."
            )

        invalid_questions = df[args.full_question_column].apply(
            lambda value: not isinstance(value, str) or not value.strip()
        )
        if invalid_questions.any():
            invalid_indices = invalid_questions[invalid_questions].index.tolist()
            raise ValueError(
                f"Found invalid questions in column '{args.full_question_column}', indices: {invalid_indices[:10]}"
            )

        if "model_answer" not in df.columns:
            df["model_answer"] = None
        if "layer_logits" not in df.columns:
            df["layer_logits"] = None
        if "raw_output" not in df.columns:
            df["raw_output"] = None

        print("Testing with first 3 questions...")
        for idx in df.index[:3]:
            question = df.at[idx, args.full_question_column]
            answer, layer_logits, raw_output = process_question(
                question=question,
                tokenizer=tokenizer,
                model=model,
                model_input_device=model_input_device,
                inference_mode=args.inference_mode,
                inference_layer=args.inference_layer,
                question_index=idx,
                answer_token_ids=answer_token_ids,
            )
            print(
                f"Test index {idx}: answer={answer}, "
                f"layers={list(layer_logits.keys())[:5]}, raw_output={raw_output[:120]!r}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("Processing all questions...")
        for idx in tqdm(df.index, total=len(df), desc="Initial processing"):
            if not is_valid_answer(df.at[idx, "model_answer"]):
                question = df.at[idx, args.full_question_column]
                answer, layer_logits, raw_output = process_question(
                    question=question,
                    tokenizer=tokenizer,
                    model=model,
                    model_input_device=model_input_device,
                    inference_mode=args.inference_mode,
                    inference_layer=args.inference_layer,
                    question_index=idx,
                    answer_token_ids=answer_token_ids,
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        retry_count = 0
        while retry_count < args.max_retries:
            invalid_indices = df.index[
                df["model_answer"].isna()
                | (df["model_answer"] == "")
                | (df["model_answer"] == "Error")
                | (~df["model_answer"].apply(is_valid_answer))
            ].tolist()

            if not invalid_indices:
                print("All entries have valid answers.")
                break

            print(
                f"Retry {retry_count + 1}/{args.max_retries}: "
                f"Found {len(invalid_indices)} invalid answers."
            )
            for idx in tqdm(invalid_indices, desc=f"Retry {retry_count + 1}"):
                question = df.at[idx, args.full_question_column]
                answer, layer_logits, raw_output = process_question(
                    question=question,
                    tokenizer=tokenizer,
                    model=model,
                    model_input_device=model_input_device,
                    inference_mode=args.inference_mode,
                    inference_layer=args.inference_layer,
                    question_index=idx,
                    answer_token_ids=answer_token_ids,
                )
                df.at[idx, "model_answer"] = answer
                df.at[idx, "layer_logits"] = layer_logits
                df.at[idx, "raw_output"] = raw_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            retry_count += 1
            time.sleep(1)

        output_dir = build_output_dir(
            dataset=args.dataset,
            question_type=args.question_type,
            prefix_type=args.prefix_type,
            prefix_subtype=args.prefix_subtype,
            academic_level=args.academic_level,
        )
        model_short_name = args.model_name.split("/")[-1].replace(".", "_")
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        inference_mode_str = "cot" if args.inference_mode == "logit_and_cot" else "logit"
        output_filename = (
            output_dir
            / f"{model_short_name}_{inference_mode_str}_{args.inference_layer}_{timestamp_str}.pkl"
        )

        invalid_count = len(
            df[
                df["model_answer"].isna()
                | (df["model_answer"] == "")
                | (df["model_answer"] == "Error")
                | (~df["model_answer"].apply(is_valid_answer))
            ]
        )
        if invalid_count > 0:
            print(
                f"Warning: {invalid_count} entries still have invalid answers "
                f"after {args.max_retries} retries."
            )
            logging.warning(
                "%s entries still have invalid answers after retries.",
                invalid_count,
            )
        else:
            print("All entries successfully populated with valid answers.")

        df.to_pickle(output_filename)
        print(f"Saved results to {output_filename}")
        logging.info("Saved mechanistic inference results to %s", output_filename)
    except Exception as exc:
        print(f"An error occurred: {exc}")
        print(traceback.format_exc())
        logging.error("An error occurred: %s\n%s", exc, traceback.format_exc())
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
