"""Helpers for layer-wise logit-lens style decoding."""

from __future__ import annotations

from typing import Sequence

import torch

from .metrics import decision_score


def decode_answer_logits_from_hidden_states(model, hidden_states: Sequence[torch.Tensor], answer_token_ids: dict[str, int]):
    lm_head = model.lm_head.weight
    final_norm = _resolve_final_norm(model)
    layer_rows: list[dict[str, float]] = []

    for layer_index, hidden in enumerate(hidden_states):
        final_token_state = hidden[:, -1, :]
        normalized = final_norm(final_token_state)
        logits = normalized @ lm_head.T
        answer_logits = {letter: float(logits[0, token_id].item()) for letter, token_id in answer_token_ids.items()}
        layer_rows.append({"layer": layer_index, **answer_logits})

    return layer_rows


def add_decision_scores(layer_rows: list[dict[str, float]], *, correct_answer: str, incorrect_answer: str) -> list[dict[str, float]]:
    enriched: list[dict[str, float]] = []
    for row in layer_rows:
        answer_values = [row["A"], row["B"], row["C"], row["D"]]
        min_logit = min(answer_values)
        max_logit = max(answer_values)
        enriched.append(
            {
                **row,
                "correct_decision_score": decision_score(row[correct_answer], min_logit, max_logit),
                "incorrect_decision_score": decision_score(row[incorrect_answer], min_logit, max_logit),
            }
        )
    return enriched


def _resolve_final_norm(model):
    if hasattr(model.model, "norm"):
        return model.model.norm
    if hasattr(model.model, "decoder") and hasattr(model.model.decoder, "final_layer_norm"):
        return model.model.decoder.final_layer_norm
    raise AttributeError("Could not find final normalization module for logit lens decoding")
