"""Closed-book QA inference for Base, ResMem, MLP Memory, Memory Decoder and LoRA.

Every method shares the concise-answer prompt, 384 input tokens, greedy
decoding of at most 12 new tokens, BF16 base computation, eager attention,
batch size 16 and full-prefix recomputation at every step. ResMem and MLP
Memory read h_t (the input of the final block's feed-forward sublayer) and run
in FP32; Memory Decoder and the LoRA-adapted base run in BF16.
"""
from __future__ import annotations

import gc
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from resmem.answer_nll import collate_encoded, encode_prompt_answer, score_log_probs
from resmem.fusion import MIXTURE_METHODS, RESIDUAL, fuse_logits
from resmem.memory import capture_memory_input, load_memory
from resmem.prompting import build_prompt

METHODS = ("base", RESIDUAL, "mlpmemory", "memory_decoder", "lora")
HIDDEN_STATE_METHODS = frozenset({RESIDUAL, "mlpmemory"})
PROMPT_PROTOCOL = "concise-format-v1"
MAX_INPUT_LENGTH = 384
MAX_NEW_TOKENS = 12
BATCH_SIZE = 16
LORA_TOPOLOGY = {"rank": 8, "alpha": 16, "target_modules": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
QUESTION_CONTINUATION_RE = re.compile(r"(?:^|\n|;\s*)question\s*:", flags=re.IGNORECASE)


def extract_first_answer_line(text: str) -> str:
    """Answer extraction used for Base, ResMem and MLP Memory."""
    stripped = text.strip()
    if not stripped:
        return ""
    line = stripped.splitlines()[0].strip()
    if "\nQuestion:" in stripped:
        line = stripped.split("\nQuestion:", 1)[0].strip()
    return line


def extract_first_answer_line_decoder(text: str) -> str:
    """Answer extraction used for Memory Decoder and LoRA."""
    stripped = text.strip()
    if not stripped:
        return ""
    match = QUESTION_CONTINUATION_RE.search(stripped)
    if match is not None:
        stripped = stripped[: match.start()].strip()
    lines = stripped.splitlines()
    return lines[0].strip() if lines else ""


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, *, rank: int, alpha: float, strength: float) -> None:
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(strength) * self.alpha / self.rank
        self.lora_a = torch.nn.Parameter(torch.zeros((self.rank, base.in_features), dtype=base.weight.dtype, device=base.weight.device))
        self.lora_b = torch.nn.Parameter(torch.zeros((base.out_features, self.rank), dtype=base.weight.dtype, device=base.weight.device))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_a), self.lora_b) * self.scaling


def _install_lora(model: torch.nn.Module, adapter_path: str, strength: float) -> list[str]:
    targets = LORA_TOPOLOGY["target_modules"]
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and any(name == t or name.endswith(f".{t}") for t in targets):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], LoRALinear(module, rank=LORA_TOPOLOGY["rank"], alpha=LORA_TOPOLOGY["alpha"], strength=strength))
            names.append(name)
    payload = torch.load(Path(adapter_path), map_location="cpu", weights_only=True)
    expected = {f"{name}.{p}" for name in names for p in ("lora_a", "lora_b")}
    if set(payload) != expected:
        raise RuntimeError("adapter checkpoint key mismatch")
    for key, tensor in payload.items():
        parameter = model.get_parameter(key)
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(f"adapter checkpoint shape mismatch: {key}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return names


def _strip_lora(model: torch.nn.Module) -> None:
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(parent, name.rsplit(".", 1)[-1], module.base)


class QARuntime:
    """Own one frozen base model and at most one memory (or adapter) at a time."""

    def __init__(self, base_model_path: str, device: str = "cuda:0", attention: str = "eager"):
        import transformers

        self.device = device
        self.attention = attention
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16, attn_implementation=attention, low_cpu_mem_usage=True
        ).to(device).eval()
        self.method = "base"
        self.memory_model = None
        self._hook = None
        self._capture: dict[str, object] = {"hidden_states": None}

    def set_method(self, method: str, checkpoint: str | None = None, lora_strength: float = 0.25) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self._release()
        self.method = method
        if method == "base":
            return
        if checkpoint is None:
            raise ValueError(f"{method} requires a checkpoint")
        if method in HIDDEN_STATE_METHODS:
            self._hook = capture_memory_input(self.base_model, self._capture)
            self.memory_model = load_memory(checkpoint, self.base_model, self.device)
        elif method == "memory_decoder":
            import transformers

            self.memory_model = transformers.AutoModelForCausalLM.from_pretrained(
                checkpoint, torch_dtype=torch.bfloat16, attn_implementation=self.attention, low_cpu_mem_usage=True
            ).to(self.device).eval()
        else:
            _install_lora(self.base_model, checkpoint, lora_strength)

    def _release(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        _strip_lora(self.base_model)
        self._capture["hidden_states"] = None
        self.memory_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract(self, text: str) -> str:
        if self.method in {"memory_decoder", "lora"}:
            return extract_first_answer_line_decoder(text)
        return extract_first_answer_line(text)

    def _memory_logits(self, ids, attention, positions=None):
        """Memory logits at the last position, or at ``positions`` when given."""
        if self.method in HIDDEN_STATE_METHODS:
            hidden = self._capture.get("hidden_states")
            if not isinstance(hidden, torch.Tensor) or hidden.shape[:2] != ids.shape:
                raise RuntimeError("final feed-forward hook did not capture h_t")
            selected = hidden[:, -1] if positions is None else hidden[positions]
            return self.memory_model(inputs_embeds=selected.float().unsqueeze(1))[:, 0].float()
        logits = self.memory_model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        return (logits[:, -1] if positions is None else logits[positions]).float()

    def _fuse(self, base_logits, memory_logits, coefficient):
        if self.method == RESIDUAL or self.method in MIXTURE_METHODS:
            return fuse_logits(base_logits, memory_logits, self.method, coefficient)
        raise ValueError(self.method)

    def _uses_memory(self, coefficient: float) -> bool:
        return self.method in HIDDEN_STATE_METHODS | MIXTURE_METHODS and float(coefficient) != 0.0

    @torch.inference_mode()
    def generate(self, rows: list[dict], coefficient: float = 0.0, batch_size: int = BATCH_SIZE) -> list[dict]:
        """Greedy EOS-or-12-token generation; ``coefficient`` is gamma (ResMem) or lambda (mixtures)."""
        tokenizer = self.tokenizer
        old_padding = tokenizer.padding_side
        tokenizer.padding_side = "left"
        results = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                prompts = [build_prompt(row["question"], prompt_protocol=PROMPT_PROTOCOL) for row in batch]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_LENGTH).to(self.device)
                ids, attention = encoded.input_ids, encoded.attention_mask
                finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=self.device)
                output_ids: list[list[int]] = [[] for _ in batch]
                for _ in range(MAX_NEW_TOKENS):
                    self._capture["hidden_states"] = None
                    base_logits = self.base_model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, -1].float()
                    if self._uses_memory(coefficient):
                        joint = self._fuse(base_logits, self._memory_logits(ids, attention), coefficient)
                    else:
                        joint = base_logits
                    next_ids = joint.argmax(dim=-1)
                    for index, token in enumerate(next_ids.tolist()):
                        if not finished[index]:
                            if int(token) == int(tokenizer.eos_token_id):
                                finished[index] = True
                            else:
                                output_ids[index].append(int(token))
                    if bool(finished.all()):
                        break
                    appended = next_ids.masked_fill(finished, int(tokenizer.pad_token_id))
                    ids = torch.cat((ids, appended[:, None]), dim=1)
                    attention = torch.cat((attention, (~finished).long()[:, None]), dim=1)
                for row, values in zip(batch, output_ids, strict=True):
                    text = tokenizer.decode(values, skip_special_tokens=True)
                    results.append({"stable_id": row["stable_id"], "prediction": self._extract(text), "output_ids": values})
        finally:
            tokenizer.padding_side = old_padding
        return results

    @torch.inference_mode()
    def score_answers(self, rows: list[dict], coefficients: list[float], batch_size: int = BATCH_SIZE) -> dict[float, list[dict]]:
        """Teacher-forced NLL of every reference-answer token (prompt and EOS excluded)."""
        encoded = [
            encode_prompt_answer(self.tokenizer, prompt=build_prompt(row["question"], prompt_protocol=PROMPT_PROTOCOL),
                                 answer=str(row["answer"]), max_length=MAX_INPUT_LENGTH)
            for row in rows
        ]
        output: dict[float, list[dict]] = {float(c): [] for c in coefficients}
        for start in range(0, len(rows), batch_size):
            batch = collate_encoded(encoded[start : start + batch_size], pad_token_id=int(self.tokenizer.pad_token_id))
            ids = batch.input_ids.to(self.device)
            attention = batch.attention_mask.to(self.device)
            positions = (batch.batch_indices.to(self.device), batch.causal_positions.to(self.device))
            targets = batch.target_ids.to(self.device)
            self._capture["hidden_states"] = None
            base_selected = self.base_model(input_ids=ids, attention_mask=attention, use_cache=False).logits[positions].float()
            memory_selected = None
            if any(self._uses_memory(c) for c in coefficients):
                memory_selected = self._memory_logits(ids, attention, positions)
            for coefficient in coefficients:
                joint = self._fuse(base_selected, memory_selected, coefficient) if self._uses_memory(coefficient) else base_selected
                nll, _, _ = score_log_probs(F.log_softmax(joint.float(), dim=-1), targets)
                values = nll.cpu().tolist()
                for row, (begin, end) in zip(rows[start : start + batch_size], batch.token_offsets, strict=True):
                    output[float(coefficient)].append({"stable_id": row["stable_id"], "token_count": end - begin, "token_nll": values[begin:end]})
        return output

