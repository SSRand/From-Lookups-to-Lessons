"""Readout rules that combine base and memory logits."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

RESIDUAL = "resmem"
MIXTURE_METHODS = frozenset({"mlpmemory", "memory_decoder"})


def fuse_logits(base_logits: torch.Tensor, memory_logits: torch.Tensor, method: str, coefficient: float) -> torch.Tensor:
    """Return fused next-token logits (or log-probabilities).

    ResMem adds the residual scores: z_B + gamma * s.
    MLP Memory and Memory Decoder interpolate probabilities:
    log((1 - lambda) p_B + lambda p_M), evaluated in float32 log space.
    """
    if base_logits.shape != memory_logits.shape:
        raise ValueError("base and memory logits must have identical shapes")
    coefficient = float(coefficient)
    if not math.isfinite(coefficient):
        raise ValueError("coefficient must be finite")
    base = base_logits.float()
    memory = memory_logits.float()
    if method == RESIDUAL:
        if coefficient < 0.0:
            raise ValueError("residual scale must be non-negative")
        return base + coefficient * memory
    if method not in MIXTURE_METHODS:
        raise ValueError(f"unknown fusion method: {method}")
    if not 0.0 <= coefficient <= 1.0:
        raise ValueError("mixture weight must lie in [0, 1]")
    if coefficient == 0.0:
        return base
    if coefficient == 1.0:
        return memory
    return torch.logaddexp(
        F.log_softmax(base, dim=-1) + math.log1p(-coefficient),
        F.log_softmax(memory, dim=-1) + math.log(coefficient),
    )
