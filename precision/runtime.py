"""Shared actual Mistral runtime; SDPA + KV cache + last-token heads."""
import math
import time
import torch
import torch.nn.functional as F
from precision.metrics import request_metrics


def parameter_inventory(model):
    counts = {}
    numel = 0
    for parameter in model.parameters():
        numel += parameter.numel()
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype,0)+parameter.numel()*parameter.element_size()
    buffers = sum(buffer.numel()*buffer.element_size() for buffer in model.buffers())
    return dict(parameter_count=numel, parameter_payload_bytes=sum(counts.values()),
        parameter_bytes_by_dtype=counts, buffer_payload_bytes=buffers)


class Runtime:
    def __init__(self, base, memory, arm, mlp_lambda=.25, res_lambda=.4):
        if arm not in ('base','mlp','res') or (arm=='base') != (memory is None):
            raise ValueError('inconsistent arm and memory')
        self.base,self.memory,self.arm = base,memory,arm
        self.mlp_lambda,self.res_lambda = mlp_lambda,res_lambda
        self.hidden = None
        self.hook = None
        self.device = next(base.parameters()).device
        if memory is not None:
            self.memory_dtype = next(memory.parameters()).dtype
            # Own only the last token, not a view retaining the complete prefill activation.
            def capture(_module,args):self.hidden=args[0][:,-1:,:].detach().clone()
            self.hook=base.model.layers[-1].mlp.register_forward_pre_hook(capture)

    @torch.inference_mode()
    def forward(self, ids, attention_mask, cache):
        positions=(attention_mask.long().cumsum(-1)-1).clamp_min(0)
        self.hidden=None
        output=self.base(input_ids=ids,attention_mask=attention_mask,
            position_ids=positions[:,-ids.shape[1]:],past_key_values=cache,
            use_cache=True,logits_to_keep=1,return_dict=True)
        logits=output.logits[:,-1,:].float()
        if self.memory is not None:
            if self.hidden is None:raise RuntimeError('memory input hook did not execute')
            memory_logits=self.memory(inputs_embeds=self.hidden.to(self.memory_dtype))[:,0,:].float()
            if self.arm=='mlp':
                logits=torch.logaddexp(F.log_softmax(logits,-1)+math.log1p(-self.mlp_lambda),
                    F.log_softmax(memory_logits,-1)+math.log(self.mlp_lambda))
            else:logits=logits+self.res_lambda*memory_logits
        self.hidden=None
        return logits,output.past_key_values

    def close(self):
        if self.hook is not None:self.hook.remove()
        self.hook=None
        self.hidden=None


@torch.inference_mode()
def generate(runtime, input_ids, attention_mask, *, max_new_tokens, fixed_tokens, eos_token_ids):
    """Model-side request includes H2D; synchronization at TTFT and final output.

    EOS is counted once as a generated token; finished padded rows are excluded
    from useful-token throughput. They remain in the static batch compute.
    """
    if max_new_tokens < 1:raise ValueError('max_new_tokens must be positive')
    device=runtime.device
    cuda=device.type=='cuda'
    if cuda:
        torch.cuda.synchronize(device)
        resident_allocated=torch.cuda.memory_allocated(device)
        resident_reserved=torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        begin_event=torch.cuda.Event(enable_timing=True)
        first_event=torch.cuda.Event(enable_timing=True)
        end_event=torch.cuda.Event(enable_timing=True)
    started=time.perf_counter()
    if cuda:begin_event.record()
    ids=input_ids.to(device)
    mask=attention_mask.to(device)
    batch=ids.shape[0]
    lengths=torch.ones(batch,dtype=torch.long,device=device)
    eos=torch.tensor(eos_token_ids,dtype=torch.long,device=device)
    logits,cache=runtime.forward(ids,mask,None)
    next_token=logits.argmax(-1)
    output_tokens=[next_token]
    finished=torch.zeros(batch,dtype=torch.bool,device=device)
    if not fixed_tokens:finished=torch.isin(next_token,eos)
    if cuda:
        first_event.record()
        torch.cuda.synchronize(device)
    first_ready=time.perf_counter()
    steps=0
    for _ in range(1,max_new_tokens):
        if not fixed_tokens and bool(finished.all()):break
        active=~finished
        ids=next_token.masked_fill(finished,0)[:,None]
        mask=torch.cat([mask,active[:,None].to(mask.dtype)],dim=-1)
        logits,cache=runtime.forward(ids,mask,cache)
        next_token=logits.argmax(-1)
        output_tokens.append(next_token.masked_fill(finished,0))
        lengths+=active.long()
        if not fixed_tokens:finished|=torch.isin(next_token,eos)
        steps+=1
    if cuda:
        end_event.record()
        torch.cuda.synchronize(device)
    ended=time.perf_counter()
    # Output transfer is intentionally outside model-side timing; full text E2E
    # is timed separately by the caller and includes this transfer/detokenization.
    lengths_cpu=lengths.tolist()
    tokens=torch.stack(output_tokens,dim=1).tolist()
    metrics=request_metrics(ttft_seconds=first_ready-started,
        decode_seconds=ended-first_ready if steps else 0,
        generated_lengths=lengths_cpu,decode_steps=steps)
    if cuda:
        metrics.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            resident_allocated_bytes=resident_allocated,resident_reserved_bytes=resident_reserved,
            cuda_ttft_seconds=begin_event.elapsed_time(first_event)/1000,
            cuda_decode_seconds=first_event.elapsed_time(end_event)/1000 if steps else 0)
    result=dict(metrics=metrics,token_ids=[row[:length] for row,length in zip(tokens,lengths_cpu)],
        stopped_on_eos=finished.tolist())
    del cache,logits
    return result
