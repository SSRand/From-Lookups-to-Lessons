"""Profiling prompts and exact shared token tensors for the timing benchmark.

Inputs repeat the tokenized source text after one BOS to exactly fill the
384-token context (all unmasked). Batch 1 uses the first row of the same tensors.
"""
from pathlib import Path
import json

PROMPTS = Path(__file__).resolve().parent / 'timing_prompts.json'
LIMITS = [('short_qa', 384)]


def freeze(tokenizer, prompts=PROMPTS):
    tokenizer.padding_side='left';tokenizer.truncation_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    source=json.loads(Path(prompts).read_text())
    frozen={'padding_side':'left','truncation_side':'left','eos_token_ids':[tokenizer.eos_token_id],'workloads':{}}
    for name,limit in LIMITS:
        rows=source['workloads'][name]
        texts=[r['text'] for r in rows]
        fixed=[]
        for text in texts:
            # Saturated compute shapes: repeated tokenized text, exact context length.
            tokens=tokenizer.encode(text,add_special_tokens=False)
            repeated=(tokens*((limit+len(tokens)-1)//len(tokens)))[:limit-1]
            fixed.append([tokenizer.bos_token_id]+repeated)
        frozen['workloads'][name]=dict(source_ids=[r['id'] for r in rows],texts=texts,max_input_tokens=limit,
            fixed_input_ids=fixed,fixed_attention_mask=[[1]*limit for _ in fixed])
    return frozen
