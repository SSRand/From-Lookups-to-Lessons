"""Reference-answer scoring: NLL of every answer token, excluding the prompt and EOS."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class EncodedExample:
    input_ids: tuple[int, ...]
    prompt_length: int
    answer_ids: tuple[int, ...]
    answer_causal_positions: tuple[int, ...]
    answer_target_ids: tuple[int, ...]


@dataclass(frozen=True)
class EncodedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    batch_indices: torch.Tensor
    causal_positions: torch.Tensor
    target_ids: torch.Tensor
    token_offsets: tuple[tuple[int, int], ...]


def encode_prompt_answer(
    tokenizer: object,
    *,
    prompt: str,
    answer: str,
    max_length: int,
) -> EncodedExample:
    if max_length <= 1:
        raise ValueError("max_length must exceed one token")
    clean_answer = answer.strip()
    if not clean_answer:
        raise ValueError("answer must be non-empty")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=True))
    answer_ids = list(tokenizer.encode(f" {clean_answer}", add_special_tokens=False))
    if not prompt_ids:
        raise ValueError("prompt tokenization is empty")
    if not answer_ids:
        raise ValueError("answer tokenization is empty")
    prompt_budget = max_length - len(answer_ids)
    if prompt_budget <= 0:
        raise ValueError("answer tokenization leaves no prompt position within max_length")
    if len(prompt_ids) > prompt_budget:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if prompt_ids[0] == bos_token_id and prompt_budget >= 1:
            prompt_ids = [prompt_ids[0]] + prompt_ids[-(prompt_budget - 1) :] if prompt_budget > 1 else [prompt_ids[0]]
        else:
            prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + answer_ids
    prompt_length = len(prompt_ids)
    causal_positions = tuple(range(prompt_length - 1, len(input_ids) - 1))
    if len(causal_positions) != len(answer_ids):
        raise AssertionError("answer causal-position construction mismatch")
    return EncodedExample(
        input_ids=tuple(int(value) for value in input_ids),
        prompt_length=prompt_length,
        answer_ids=tuple(int(value) for value in answer_ids),
        answer_causal_positions=causal_positions,
        answer_target_ids=tuple(int(value) for value in answer_ids),
    )


def collate_encoded(
    encoded_examples: Sequence[EncodedExample], *, pad_token_id: int
) -> EncodedBatch:
    if not encoded_examples:
        raise ValueError("cannot collate an empty batch")
    max_length = max(len(example.input_ids) for example in encoded_examples)
    input_ids = torch.full(
        (len(encoded_examples), max_length),
        fill_value=int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    batch_indices: list[int] = []
    causal_positions: list[int] = []
    target_ids: list[int] = []
    token_offsets: list[tuple[int, int]] = []
    offset = 0
    for batch_index, example in enumerate(encoded_examples):
        sequence_length = len(example.input_ids)
        input_ids[batch_index, :sequence_length] = torch.tensor(
            example.input_ids, dtype=torch.long
        )
        attention_mask[batch_index, :sequence_length] = 1
        token_count = len(example.answer_target_ids)
        if token_count != len(example.answer_causal_positions):
            raise ValueError("encoded answer positions and targets must align")
        if any(position < 0 or position >= sequence_length - 1 for position in example.answer_causal_positions):
            raise ValueError("answer causal position lies outside the unpadded sequence")
        batch_indices.extend([batch_index] * token_count)
        causal_positions.extend(example.answer_causal_positions)
        target_ids.extend(example.answer_target_ids)
        token_offsets.append((offset, offset + token_count))
        offset += token_count
    return EncodedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_indices=torch.tensor(batch_indices, dtype=torch.long),
        causal_positions=torch.tensor(causal_positions, dtype=torch.long),
        target_ids=torch.tensor(target_ids, dtype=torch.long),
        token_offsets=tuple(token_offsets),
    )


def score_log_probs(
    log_probs: torch.Tensor, target_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [tokens, vocab]")
    if target_ids.ndim != 1 or target_ids.shape[0] != log_probs.shape[0]:
        raise ValueError("target_ids must align with the token dimension")
    if target_ids.numel() == 0:
        raise ValueError("cannot score zero target tokens")
    if int(target_ids.min()) < 0 or int(target_ids.max()) >= log_probs.shape[-1]:
        raise ValueError("target token id is outside the vocabulary")
    row_indices = torch.arange(target_ids.shape[0], device=log_probs.device)
    gold_log_probs = log_probs[row_indices, target_ids]
    nll = -gold_log_probs
    ranks = (log_probs > gold_log_probs.unsqueeze(-1)).sum(dim=-1) + 1
    top1 = log_probs.argmax(dim=-1).eq(target_ids)
    return nll, ranks, top1
