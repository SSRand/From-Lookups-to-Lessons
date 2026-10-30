"""Pack the ranked top-5 retrieved passages into the concise RAG prompt (4,096-token input cap).

    python -m qa.baselines.bm25_rag.prepare_prompts --test-root data/qa/test/frozen \
        --retrieval runs/bm25_rag/retrieval --output runs/bm25_rag/prepared.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path
from qa.baselines.bm25_rag import rag
from qa.common import DATASETS, load_rows
from resmem.scoring import row_aliases


def complete_top5(source, visible):
    norm = lambda s: ' '.join(str(s).split())
    return (len(source) == 5 and len(visible) == 5 and len({str(d['doc_id']) for d in source}) == 5
            and all(str(o['doc_id']) == str(p['doc_id']) and norm(o['text']) == norm(p['text']) for o, p in zip(source, visible)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test-root', type=Path, required=True)
    p.add_argument('--retrieval', type=Path, required=True)
    p.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.base_model)
    position = 0; stats = {}; seen = set()
    partial = a.output.with_name(a.output.name + '.partial')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open('w') as out:
        for ds in DATASETS:
            refs = load_rows(a.test_root/f'{ds}.jsonl'); retrieved = load_rows(a.retrieval/f'{ds}__bm25_rag.retrieval.jsonl')
            if len(refs) != len(retrieved): raise ValueError('Wrong dataset coverage')
            lengths = []; visible_answers = 0; complete = 0; truncated_ids = []
            for row, source in zip(refs, retrieved, strict=True):
                if row['stable_id'] != source['stable_id'] or row['question'] != source['question'] or row['dataset'] != ds:
                    raise ValueError('Retrieval identity changed')
                if row['stable_id'] in seen: raise ValueError('Duplicate frozen QA ID')
                seen.add(row['stable_id'])
                docs = source['retrieved_documents'][:5]
                packed = rag.pack_context_prompt(question=row['question'],
                    retrieved_documents=[rag.Document(str(d['doc_id']), str(d['text'])) for d in docs],
                    tokenizer=tokenizer, max_input_length=4160, max_new_tokens=64, prompt_top_k=5, max_document_tokens=4096)
                visible = [dict(doc_id=d.doc_id, text=d.text) for d in packed['visible_documents']]
                n = len(tokenizer.encode(packed['prompt'], add_special_tokens=True))
                if n > 4096: raise ValueError('Full top5 exceeds input budget')
                intact = complete_top5(docs, visible)
                complete += int(intact)
                if not intact: truncated_ids.append(row['stable_id'])
                hit = rag.visible_answer_hit(row_aliases(row), packed['visible_documents'])
                record = dict(index=position, stable_id=row['stable_id'], dataset=ds, reference=row, prompt=packed['prompt'],
                    prompt_tokens=n, prompt_sha256=hashlib.sha256(packed['prompt'].encode()).hexdigest(),
                    visible_documents=visible, visible_answer=hit, complete_top5=intact)
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
                lengths.append(n); visible_answers += hit; position += 1
            stats[ds] = dict(rows=len(lengths), min=min(lengths), max=max(lengths), median=sorted(lengths)[len(lengths)//2], mean=sum(lengths)/len(lengths),
                             visible_answer_fraction=visible_answers/len(lengths), all_top5_preserved=complete,
                             truncated_top5_count=len(truncated_ids), truncated_stable_ids=truncated_ids)
            print(json.dumps(dict(stage='prepare', dataset=ds, rows=position, statistics=stats[ds])), flush=True)
    partial.replace(a.output)
    a.output.with_name(a.output.stem + '.stats.json').write_text(json.dumps(stats, indent=2)+'\n')


if __name__ == '__main__':
    main()
