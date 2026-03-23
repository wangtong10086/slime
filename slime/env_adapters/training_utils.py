from __future__ import annotations

import json
from typing import Any

from slime.utils.toolcall_health import extract_assistant_response_text, preserve_structured_conversation
from slime.utils.types import Sample


def normalize_conversation_for_training(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return preserve_structured_conversation(conversation)


def _normalize_token_ids(encoded: Any) -> list[int]:
    if isinstance(encoded, dict):
        encoded = encoded.get("input_ids", encoded)
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def build_training_tokens_and_mask(tokenizer, conversation: list[dict[str, Any]]) -> tuple[list[int], int, list[int]]:
    tools = conversation[0].get("tools") if conversation else None
    messages = []
    full_mask: list[int] = []
    previous_len = 0

    for message in conversation:
        msg = dict(message)
        msg.pop("tools", None)
        messages.append(msg)
        current_ids = _normalize_token_ids(
            tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        delta = len(current_ids) - previous_len
        if delta < 0:
            raise ValueError("Tokenized conversation length decreased unexpectedly")
        mask_value = 1 if msg.get("role") == "assistant" else 0
        full_mask.extend([mask_value] * delta)
        previous_len = len(current_ids)

    full_tokens = _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if len(full_tokens) != len(full_mask):
        raise ValueError(f"Token/mask length mismatch: {len(full_tokens)} != {len(full_mask)}")

    try:
        first_supervised_idx = full_mask.index(1)
    except ValueError:
        first_supervised_idx = len(full_tokens)

    response_length = len(full_tokens) - first_supervised_idx
    loss_mask = full_mask[first_supervised_idx:]
    return full_tokens, response_length, loss_mask


def build_sample_from_conversation(
    *,
    sample: Sample,
    tokenizer,
    conversation: list[dict[str, Any]],
    reward: float,
    metadata: dict[str, Any],
    fallback_payload: str,
) -> Sample:
    normalized = normalize_conversation_for_training(conversation)
    full_tokens, response_length, loss_mask = build_training_tokens_and_mask(tokenizer, normalized)
    if response_length <= 0:
        normalized = [*normalized, {"role": "assistant", "content": fallback_payload}]
        full_tokens, response_length, loss_mask = build_training_tokens_and_mask(tokenizer, normalized)

    sample.tokens = full_tokens
    sample.response_length = response_length
    sample.loss_mask = loss_mask
    sample.response = extract_assistant_response_text(conversation)
    sample.reward = reward
    sample.prompt = normalized
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {**(sample.metadata or {}), **metadata}
    return sample
