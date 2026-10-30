"""Memory Decoder arm for the shared SDPA + KV-cache cost runtime.

Mirrors precision.runtime.Runtime.forward: BF16 Base with mask-derived position
IDs, last-token logits, FP32 fusion. The MD is a separate causal LM over the
same token stream and keeps its own KV cache; fusion is the FP32 log-space
probability mixture used for Memory Decoder in qa.runtime.
"""
import math
import torch
import torch.nn.functional as F


class MDRuntime:
    def __init__(self, base, memory, lam=.25):
        if memory is None or not 0 < lam < 1: raise ValueError('MD arm requires a memory and 0<lambda<1')
        self.base, self.memory, self.lam, self.arm = base, memory, float(lam), 'md'
        self.device = next(base.parameters()).device
        self.memory_dtype = next(memory.parameters()).dtype

    @torch.inference_mode()
    def forward(self, ids, attention_mask, cache):
        base_cache, md_cache = (None, None) if cache is None else cache
        positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1]:]
        out = self.base(input_ids=ids, attention_mask=attention_mask, position_ids=positions,
                        past_key_values=base_cache, use_cache=True, logits_to_keep=1, return_dict=True)
        mem = self.memory(input_ids=ids, attention_mask=attention_mask, position_ids=positions,
                          past_key_values=md_cache, use_cache=True, logits_to_keep=1, return_dict=True)
        logits = torch.logaddexp(F.log_softmax(out.logits[:, -1, :].float(), -1) + math.log1p(-self.lam),
                                 F.log_softmax(mem.logits[:, -1, :].float(), -1) + math.log(self.lam))
        return logits, (out.past_key_values, mem.past_key_values)

    def close(self):
        pass
