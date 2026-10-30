"""Greedy full-prefix generation of the original QA harness (trimmed to the functions used)."""
from __future__ import annotations

import math
import time


SYSTEMS = ("base", "mlpmemory", "residual")


def _is_torch_tensor(value: object) -> bool:
    return value.__class__.__module__.split(".", 1)[0] == "torch"


def fuse_next_token_logits(
    base_logits,
    memory_logits,
    *,
    method: str,
    lambda_: float,
    residual_scale: float,
):
    """Apply the frozen MLPMemory or RES inference fusion rule."""

    if getattr(base_logits, "shape", None) != getattr(memory_logits, "shape", None):
        raise ValueError("base and memory logits must have identical shapes")
    if method == "mlpmemory":
        if not 0.0 < lambda_ < 1.0 or not math.isfinite(lambda_):
            raise ValueError("MLPMemory lambda must lie strictly inside (0, 1)")
        if _is_torch_tensor(base_logits):
            import torch
            import torch.nn.functional as F

            return torch.logaddexp(
                F.log_softmax(base_logits.float(), dim=-1) + math.log1p(-lambda_),
                F.log_softmax(memory_logits.float(), dim=-1) + math.log(lambda_),
            )
        import numpy as np

        base_shifted = base_logits - np.max(base_logits, axis=-1, keepdims=True)
        memory_shifted = memory_logits - np.max(memory_logits, axis=-1, keepdims=True)
        base_probabilities = np.exp(base_shifted)
        base_probabilities /= base_probabilities.sum(axis=-1, keepdims=True)
        memory_probabilities = np.exp(memory_shifted)
        memory_probabilities /= memory_probabilities.sum(axis=-1, keepdims=True)
        return np.log(
            (1.0 - lambda_) * base_probabilities
            + lambda_ * memory_probabilities
        )
    if method == "residual":
        if lambda_ < 0.0 or not math.isfinite(lambda_):
            raise ValueError("RES lambda must be finite and non-negative")
        if residual_scale <= 0.0 or not math.isfinite(residual_scale):
            raise ValueError("residual scale must be finite and positive")
        return base_logits + lambda_ * residual_scale * memory_logits
    raise ValueError(f"unknown memory method: {method}")


def extract_first_answer_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0].strip()
    if "\nQuestion:" in stripped:
        line = stripped.split("\nQuestion:", 1)[0].strip()
    return line


def _generate_predictions(
    *,
    base_model,
    memory_model,
    tokenizer,
    examples,
    method: str,
    lambda_: float,
    residual_scale: float,
    capture: dict[str, object],
    max_new_tokens: int,
    max_input_length: int,
    batch_size: int,
    device,
) -> list[str]:
    import torch

    if method not in SYSTEMS:
        raise ValueError(f"unknown generation method: {method}")
    if method != "base" and memory_model is None:
        raise ValueError(f"{method} generation requires a memory model")
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"
    stop_ids = {int(tokenizer.eos_token_id)}
    predictions: list[str] = []
    started = time.monotonic()
    try:
        for start in range(0, len(examples), batch_size):
            current = examples[start : start + batch_size]
            encoded = tokenizer(
                [value.prompt for value in current],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
            ids = encoded.input_ids
            attention = encoded.attention_mask
            finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=device)
            output_ids: list[list[int]] = [[] for _ in current]
            for _ in range(max_new_tokens):
                with torch.inference_mode():
                    capture["hidden_states"] = None
                    base_logits = base_model(
                        input_ids=ids,
                        attention_mask=attention,
                        use_cache=False,
                    ).logits[:, -1].float()
                    if method == "base":
                        joint = base_logits
                    else:
                        hidden = capture.get("hidden_states")
                        if not isinstance(hidden, torch.Tensor) or hidden.shape[:2] != ids.shape:
                            raise RuntimeError("final MLP hook did not capture the generation hidden state")
                        memory_logits = memory_model(inputs_embeds=hidden[:, -1].float().unsqueeze(1))[:, 0].float()
                        joint = fuse_next_token_logits(
                            base_logits,
                            memory_logits,
                            method=method,
                            lambda_=lambda_,
                            residual_scale=residual_scale,
                        )
                    next_ids = joint.argmax(dim=-1)
                for index, token in enumerate(next_ids.tolist()):
                    if not finished[index]:
                        if int(token) in stop_ids:
                            finished[index] = True
                        else:
                            output_ids[index].append(int(token))
                if bool(finished.all()):
                    break
                appended = next_ids.masked_fill(finished, int(tokenizer.pad_token_id))
                ids = torch.cat((ids, appended[:, None]), dim=1)
                attention = torch.cat((attention, (~finished).long()[:, None]), dim=1)
            predictions.extend(
                extract_first_answer_line(tokenizer.decode(value, skip_special_tokens=True))
                for value in output_ids
            )
            completed = min(start + batch_size, len(examples))
            if completed == len(examples) or completed % max(100, batch_size) == 0:
                print(
                    f"[odqa-generation] method={method} {completed}/{len(examples)} "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )
    finally:
        tokenizer.padding_side = old_padding
    return predictions
