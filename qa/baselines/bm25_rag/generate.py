"""BM25-RAG generation: Mistral-7B-v0.3, BF16/eager/greedy/KV cache, left padding, batch 8, at most 64 new tokens.

Prepared rows are split into 8 fixed slots (``index % 8``) decoded in batches of
8, so batch composition does not depend on how many processes run. Launch one
process per GPU; ``collect`` then writes qa.common prediction cells
(coefficient 0) that qa.evaluate reads.

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m qa.baselines.bm25_rag.generate generate \
        --prepared runs/bm25_rag/prepared.jsonl --output runs/bm25_rag/test --shard 0 --num-shards 8
    python -m qa.baselines.bm25_rag.generate collect --prepared runs/bm25_rag/prepared.jsonl --output runs/bm25_rag/test
"""
import argparse
import json
import re
from pathlib import Path
from qa.common import DATASETS, cell_path, load_rows, write_jsonl
from resmem.scoring import normalized_metrics, row_aliases

SLOTS = 8
QUESTION_CONTINUATION_RE = re.compile(r"(?:^|\n|;\s*)question\s*:", flags=re.IGNORECASE)


def extract_first_answer_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    match = QUESTION_CONTINUATION_RE.search(stripped)
    if match is not None:
        stripped = stripped[: match.start()].strip()
    lines = stripped.splitlines()
    return lines[0].strip() if lines else ""


def generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    torch.set_num_threads(4)
    prepared = load_rows(a.prepared)
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    tokenizer.pad_token = tokenizer.eos_token; tokenizer.padding_side = 'left'
    config = GenerationConfig(max_new_tokens=64, do_sample=False, num_beams=1, use_cache=True,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0, min_length=0, min_new_tokens=0, forced_bos_token_id=None, forced_eos_token_id=None)
    model = None
    for slot in range(a.shard, SLOTS, a.num_shards):
        rows = [r for r in prepared if r['index'] % SLOTS == slot]
        path = a.output/'slots'/f'slot-{slot}.jsonl'; path.parent.mkdir(parents=True, exist_ok=True)
        done = [json.loads(line) for line in path.open() if line.endswith('\n')] if path.exists() else []
        # Resume only at a complete batch boundary.
        processed = len(done) - len(done) % 8 if len(done) < len(rows) else len(done)
        if processed == len(rows): continue
        if [r['stable_id'] for r in done[:processed]] != [r['stable_id'] for r in rows[:processed]]: raise ValueError('Resume prefix changed')
        write_jsonl(path, done[:processed])
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(a.base_model, torch_dtype=torch.bfloat16,
                attn_implementation='eager', low_cpu_mem_usage=True).to(a.device).eval()
        with path.open('a', buffering=1) as handle:
            for start in range(processed, len(rows), 8):
                batch = rows[start:start+8]
                inputs = tokenizer([r['prompt'] for r in batch], return_tensors='pt', padding=True, truncation=False).to(a.device)
                width = inputs['input_ids'].shape[1]
                if width > 4096: raise ValueError('Actual model input exceeds cap')
                with torch.inference_mode(): generated = model.generate(**inputs, generation_config=config)
                for row, tokens in zip(batch, generated[:, width:].tolist(), strict=True):
                    if tokenizer.eos_token_id in tokens: tokens = tokens[:tokens.index(tokenizer.eos_token_id)+1]
                    raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    prediction = extract_first_answer_line(raw)
                    metrics = normalized_metrics(prediction, row_aliases(row['reference']))
                    handle.write(json.dumps(dict(index=row['index'], stable_id=row['stable_id'], dataset=row['dataset'],
                        prediction=prediction, raw_prediction=raw, generated_token_ids=tokens, metrics=metrics,
                        visible_answer=row['visible_answer'], prompt_sha256=row['prompt_sha256']), ensure_ascii=False) + '\n')
                print(json.dumps({'slot': slot, 'processed': start+len(batch), 'total': len(rows)}), flush=True)
                del inputs, generated


def collect(a):
    prepared = load_rows(a.prepared)
    records = {}
    for slot in range(SLOTS):
        for r in load_rows(a.output/'slots'/f'slot-{slot}.jsonl'):
            if r['index'] % SLOTS != slot or r['stable_id'] in records: raise ValueError('Foreign or duplicate prediction')
            records[r['stable_id']] = r
    if set(records) != {r['stable_id'] for r in prepared}: raise ValueError('Incomplete population')
    for ds in DATASETS:
        rows = [records[r['stable_id']] for r in prepared if r['dataset'] == ds]
        write_jsonl(cell_path(a.output, ds, 0.0, 'predictions'),
                    [dict(stable_id=r['stable_id'], prediction=r['prediction'], output_ids=r['generated_token_ids']) for r in rows])
        m = {k: sum(r['metrics'][k] for r in rows)/len(rows) for k in ('exact_match', 'token_f1')}
        print(json.dumps(dict(dataset=ds, rows=len(rows), **m, visible_answer_fraction=sum(r['visible_answer'] for r in rows)/len(rows))))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=['generate', 'collect'])
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    a = p.parse_args()
    generate(a) if a.action == 'generate' else collect(a)


if __name__ == '__main__':
    main()
