"""BM25-RAG reference-answer NLL: the Base conditions on the exact packed RAG prompt.

Every canonical-answer token is scored (no EOS target, no generation cap) in
batches of 4 with a per-dataset input cap of 4,096 plus the longest answer, so
the prompt is never truncated. Writes qa.common NLL cells (coefficient 0) next
to the generated predictions; one process per GPU shards over datasets.

    python -m qa.baselines.bm25_rag.score_answers --prepared runs/bm25_rag/prepared.jsonl \
        --output runs/bm25_rag/test --shard 0 --num-shards 5
"""
import argparse
import json
from pathlib import Path
from qa.common import DATASETS, cell_path, load_rows, write_jsonl

BATCH = 4


def score_dataset(rt, rows):
    import torch
    import torch.nn.functional as F
    from resmem.answer_nll import collate_encoded, encode_prompt_answer, score_log_probs
    tok = rt.tokenizer
    cap = 4096 + max(len(tok.encode(' '+r['reference']['answer'].strip(), add_special_tokens=False)) for r in rows)
    encoded = []
    for r in rows:
        enc = encode_prompt_answer(tok, prompt=r['prompt'], answer=str(r['reference']['answer']), max_length=cap)
        prompt_ids = tok.encode(r['prompt'], add_special_tokens=True)
        if list(enc.input_ids[:enc.prompt_length]) != prompt_ids or len(prompt_ids) != r['prompt_tokens'] or len(prompt_ids) > 4096:
            raise ValueError('RAG conditioning truncated')
        encoded.append(enc)
    output = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch = collate_encoded(encoded[start:start+BATCH], pad_token_id=int(tok.pad_token_id))
            ids = batch.input_ids.to(rt.device); attention = batch.attention_mask.to(rt.device)
            positions = (batch.batch_indices.to(rt.device), batch.causal_positions.to(rt.device))
            logits = rt.base_model(input_ids=ids, attention_mask=attention, use_cache=False).logits[positions].float()
            nll, _, _ = score_log_probs(F.log_softmax(logits, dim=-1), batch.target_ids.to(rt.device))
            values = nll.cpu().tolist()
            for r, (begin, end) in zip(rows[start:start+BATCH], batch.token_offsets, strict=True):
                output.append({'stable_id': r['stable_id'], 'token_count': end-begin, 'token_nll': values[begin:end]})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    a = p.parse_args()
    mine = [ds for i, ds in enumerate(DATASETS) if i % a.num_shards == a.shard and not cell_path(a.output, ds, 0.0, 'nll').exists()]
    if not mine: return
    from qa.runtime import QARuntime
    rt = QARuntime(a.base_model)
    prepared = load_rows(a.prepared)
    for ds in mine:
        rows = [r for r in prepared if r['dataset'] == ds]
        write_jsonl(cell_path(a.output, ds, 0.0, 'nll'), score_dataset(rt, rows))
        print(json.dumps({'dataset': ds, 'rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
