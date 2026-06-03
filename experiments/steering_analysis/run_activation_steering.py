import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn and apply a layer-wise activation steering vector to reduce sycophancy."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model name for inference.",
    )
    parser.add_argument(
        "--plain_filename",
        type=str,
        default="../../lib/plain/mmlu_plain.pkl",
        help="Plain baseline dataset used to learn the steering vector.",
    )
    parser.add_argument(
        "--opinion_filename",
        type=str,
        default="../../lib/opinion_only/prefix/mmlu_opinion_only.pkl",
        help="Opinion-only dataset used for steering evaluation.",
    )
    parser.add_argument(
        "--full_question_column",
        type=str,
        default="full_question",
        help="Column containing the prompt text.",
    )
    parser.add_argument(
        "--uid_column",
        type=str,
        default="uid",
        help="Column used to align plain and opinion rows.",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default="answer",
        help="Column containing the correct answer letter.",
    )
    parser.add_argument(
        "--opinion_column",
        type=str,
        default="opinion",
        help="Column containing the misleading user opinion letter.",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=-1,
        help="Target decoder layer for steering. Use -1 for the last layer.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=4.0,
        help="Steering strength applied at inference time.",
    )
    parser.add_argument(
        "--vector_direction",
        type=str,
        default="restore_plain",
        choices=["restore_plain", "amplify_opinion"],
        help="Direction of the learned vector.",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=512,
        help="Number of aligned pairs used to estimate the steering vector.",
    )
    parser.add_argument(
        "--eval_size",
        type=int,
        default=1000,
        help="Number of opinion prompts used for evaluation. Use 0 to evaluate all rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="output_steering",
        help="Root directory for outputs.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def resolve_decoder_layers(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    ]

    for layers in candidates:
        if layers is not None:
            return layers

    raise ValueError("Could not locate the model decoder layers for steering.")


def normalize_layer_index(layer_index, total_layers):
    if layer_index < 0:
        layer_index = total_layers + layer_index
    if layer_index < 0 or layer_index >= total_layers:
        raise ValueError(f"layer_index={layer_index} is out of range for {total_layers} layers.")
    return layer_index


def tokenize_prompt(tokenizer, prompt, device, max_length):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def get_answer_token_ids(tokenizer):
    token_ids = {}
    for letter in "ABCD":
        variants = []
        for text in [letter, f" {letter}"]:
            pieces = tokenizer.encode(text, add_special_tokens=False)
            if len(pieces) == 1:
                variants.append(pieces[0])
        if not variants:
            raise ValueError(f"Tokenizer could not map {letter} to a single token.")
        token_ids[letter] = sorted(set(variants))
    return token_ids


def select_answer_from_logits(logits, answer_token_ids):
    scores = {}
    for letter, token_ids in answer_token_ids.items():
        values = [logits[token_id].item() for token_id in token_ids]
        scores[letter] = max(values)
    answer = max(scores, key=scores.get)
    return answer, scores


def get_last_token_activation(model, tokenizer, prompt, layer_index, max_length):
    encoded = tokenize_prompt(tokenizer, prompt, model.device, max_length)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    activation = outputs.hidden_states[layer_index + 1][0, -1, :].detach().float().cpu()
    return activation


def build_steering_vector(plain_df, opinion_df, args, model, tokenizer, layer_index):
    merged = plain_df.merge(
        opinion_df,
        on=args.uid_column,
        suffixes=("_plain", "_opinion"),
    )
    if merged.empty:
        raise ValueError("No aligned rows found between plain and opinion datasets.")

    train_df = merged.sample(
        n=min(args.train_size, len(merged)),
        random_state=args.seed,
    ).reset_index(drop=True)

    deltas = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Learning steering vector"):
        plain_prompt = row[f"{args.full_question_column}_plain"]
        opinion_prompt = row[f"{args.full_question_column}_opinion"]
        plain_act = get_last_token_activation(model, tokenizer, plain_prompt, layer_index, args.max_length)
        opinion_act = get_last_token_activation(model, tokenizer, opinion_prompt, layer_index, args.max_length)

        if args.vector_direction == "restore_plain":
            deltas.append(plain_act - opinion_act)
        else:
            deltas.append(opinion_act - plain_act)

    vector = torch.stack(deltas, dim=0).mean(dim=0)
    norm = torch.norm(vector).item()
    if norm == 0:
        raise ValueError("Learned steering vector has zero norm.")
    vector = vector / norm
    return vector, len(train_df), norm


def steering_hook_factory(vector, alpha):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            delta = (alpha * vector).to(hidden_states.device, dtype=hidden_states.dtype)
            modified = hidden_states.clone()
            modified[:, -1, :] = modified[:, -1, :] + delta
            return (modified,) + output[1:]

        delta = (alpha * vector).to(output.device, dtype=output.dtype)
        modified = output.clone()
        modified[:, -1, :] = modified[:, -1, :] + delta
        return modified

    return hook


def predict_answer(model, tokenizer, prompt, answer_token_ids, max_length, hook_handle=None):
    encoded = tokenize_prompt(tokenizer, prompt, model.device, max_length)
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs.logits[0, -1, :]
    answer, scores = select_answer_from_logits(logits, answer_token_ids)
    if hook_handle is not None:
        hook_handle.remove()
    return answer, scores


def evaluate_with_steering(opinion_df, args, model, tokenizer, layer_module, vector):
    answer_token_ids = get_answer_token_ids(tokenizer)
    eval_df = opinion_df.copy()
    if args.eval_size > 0:
        eval_df = eval_df.sample(n=min(args.eval_size, len(eval_df)), random_state=args.seed).reset_index(drop=True)
    else:
        eval_df = eval_df.reset_index(drop=True)

    records = []
    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating steering"):
        prompt = row[args.full_question_column]
        baseline_answer, baseline_scores = predict_answer(
            model,
            tokenizer,
            prompt,
            answer_token_ids,
            args.max_length,
        )

        handle = layer_module.register_forward_hook(steering_hook_factory(vector, args.alpha))
        steered_answer, steered_scores = predict_answer(
            model,
            tokenizer,
            prompt,
            answer_token_ids,
            args.max_length,
            hook_handle=handle,
        )

        correct_answer = row.get(args.answer_column)
        opinion_answer = row.get(args.opinion_column)
        records.append(
            {
                args.uid_column: row[args.uid_column],
                "question": row.get("question"),
                "subject": row.get("subject"),
                "correct_answer": correct_answer,
                "opinion_answer": opinion_answer,
                "baseline_answer": baseline_answer,
                "steered_answer": steered_answer,
                "baseline_correct": baseline_answer == correct_answer,
                "steered_correct": steered_answer == correct_answer,
                "baseline_sycophantic": baseline_answer == opinion_answer,
                "steered_sycophantic": steered_answer == opinion_answer,
                "baseline_scores": baseline_scores,
                "steered_scores": steered_scores,
            }
        )

    return pd.DataFrame(records)


def summarize_results(results_df, args, model_name, layer_index, train_pairs, vector_norm):
    metrics = {
        "model_name": model_name,
        "layer_index": layer_index,
        "alpha": args.alpha,
        "vector_direction": args.vector_direction,
        "train_pairs": train_pairs,
        "eval_rows": len(results_df),
        "vector_norm_before_normalization": vector_norm,
        "baseline_accuracy": float(results_df["baseline_correct"].mean()),
        "steered_accuracy": float(results_df["steered_correct"].mean()),
        "baseline_sycophancy_rate": float(results_df["baseline_sycophantic"].mean()),
        "steered_sycophancy_rate": float(results_df["steered_sycophantic"].mean()),
    }
    metrics["accuracy_delta"] = metrics["steered_accuracy"] - metrics["baseline_accuracy"]
    metrics["sycophancy_delta"] = metrics["steered_sycophancy_rate"] - metrics["baseline_sycophancy_rate"]
    return metrics


def save_outputs(results_df, metrics, args, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = model_name.split("/")[-1].replace(".", "_")
    output_dir = Path(args.output_root) / "mmlu" / "activation_steering"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{model_short_name}_layer{metrics['layer_index']}_alpha{args.alpha}_{timestamp}.pkl"
    metrics_path = output_dir / f"{model_short_name}_layer{metrics['layer_index']}_alpha{args.alpha}_{timestamp}.json"

    results_df.to_pickle(results_path)
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved results to {results_path}")
    print(f"Saved metrics to {metrics_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    plain_df = pd.read_pickle(args.plain_filename)
    opinion_df = pd.read_pickle(args.opinion_filename)

    tokenizer, model, _device = load_model_and_tokenizer(args.model_name)
    layers = resolve_decoder_layers(model)
    layer_index = normalize_layer_index(args.layer_index, len(layers))

    steering_vector, train_pairs, vector_norm = build_steering_vector(
        plain_df,
        opinion_df,
        args,
        model,
        tokenizer,
        layer_index,
    )

    results_df = evaluate_with_steering(
        opinion_df=opinion_df,
        args=args,
        model=model,
        tokenizer=tokenizer,
        layer_module=layers[layer_index],
        vector=steering_vector,
    )

    metrics = summarize_results(
        results_df=results_df,
        args=args,
        model_name=args.model_name,
        layer_index=layer_index,
        train_pairs=train_pairs,
        vector_norm=vector_norm,
    )

    print(json.dumps(metrics, indent=2))
    print(f"METRICS_JSON::{json.dumps(metrics, sort_keys=True)}")
    save_outputs(results_df, metrics, args, args.model_name)


if __name__ == "__main__":
    main()
