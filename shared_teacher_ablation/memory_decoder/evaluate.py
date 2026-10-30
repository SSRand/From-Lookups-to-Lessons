"""Evaluate shared-teacher Memory Decoder checkpoints with the ablation's population and runtime.

Evaluation only; Memory Decoder checkpoints are trained elsewhere. Runtime: BF16 Base and BF16
Memory Decoder, eager attention, no KV cache (full prefix every step), left padding, 384 input
tokens, 12 greedy new tokens, batches of 16 within dataset (as eval.outputs.batches), mixture
(1-lambda) p_B + lambda p_MD in float32 log space. Dev (1,024 NQ/TriviaQA) selects one lambda per
seed from the ablation grid (EM, then F1, then smaller lambda); test = all 38,627 questions at that
lambda, with complete gold-answer NLL (Memory Decoder run on the full prompt+answer sequence).

  python -m shared_teacher_ablation.memory_decoder.evaluate --population-root POP --base BASE \\
      --checkpoint-template 'MD/s{seed}' --output MDOUT [--gpus 0,1,...]
  python -m shared_teacher_ablation.memory_decoder.evaluate --summarize --population-root POP --output MDOUT
"""
import argparse,json,os,re,sys,time
from pathlib import Path
import numpy as np
GRID=(0.,.1,.2,.3,.4,.5,.75,1.)
SEEDS=(42,123,456,789,2026,2027,2028,2029)
QUESTION_CONTINUATION_RE=re.compile(r"(?:^|\n|;\s*)question\s*:",flags=re.IGNORECASE)
REPO=Path(__file__).resolve().parents[2]

def jsonl(p):
    with open(p) as f: return [json.loads(x) for x in f if x.strip()]
def batches(rr):
    out, batch, ds = [], [], None
    for r in rr:
        if batch and (len(batch) == 16 or r['dataset'] != ds): out.append(batch); batch = []
        batch.append(r); ds = r['dataset']
    if batch: out.append(batch)
    return out

def extract_first_answer_line(text):
    stripped = text.strip()
    if not stripped:
        return ""
    match = QUESTION_CONTINUATION_RE.search(stripped)
    if match is not None:
        stripped = stripped[: match.start()].strip()
    lines = stripped.splitlines()
    return lines[0].strip() if lines else ""

class Runtime:
    def __init__(self, base_path, checkpoint, device='cuda'):
        import torch, transformers
        self.torch = torch; self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(base_path)
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        load = lambda path: transformers.AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='eager', low_cpu_mem_usage=True).to(device).eval()
        self.base = load(base_path); self.memory = load(checkpoint)

    def generate(self, rows, lam):
        from resmem.fusion import fuse_logits
        from resmem.prompting import build_prompt
        torch, tok = self.torch, self.tokenizer
        old_padding = tok.padding_side; tok.padding_side = 'left'
        try:
            encoded = tok([build_prompt(r['question'], prompt_protocol='concise-format-v1') for r in rows], return_tensors='pt', padding=True, truncation=True, max_length=384).to(self.device)
            ids, attention = encoded.input_ids, encoded.attention_mask
            finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=self.device)
            output_ids = [[] for _ in rows]
            with torch.inference_mode():
                for _ in range(12):
                    base_logits = self.base(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, -1].float()
                    if lam == 0.0:
                        joint = base_logits
                    else:
                        memory_logits = self.memory(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, -1].float()
                        joint = fuse_logits(base_logits, memory_logits, 'memory_decoder', lam)
                    next_ids = joint.argmax(dim=-1)
                    for index, token in enumerate(next_ids.tolist()):
                        if not bool(finished[index]):
                            if int(token) == int(tok.eos_token_id): finished[index] = True
                            else: output_ids[index].append(int(token))
                    if bool(finished.all()): break
                    appended = next_ids.masked_fill(finished, int(tok.pad_token_id))
                    ids = torch.cat((ids, appended[:, None]), dim=1)
                    attention = torch.cat((attention, (~finished).long()[:, None]), dim=1)
        finally:
            tok.padding_side = old_padding
        return [dict(output_ids=value, prediction=extract_first_answer_line(tok.decode(value, skip_special_tokens=True))) for value in output_ids]

    def gold_nll(self, rows, lam):
        from resmem.answer_nll import encode_prompt_answer, collate_encoded, score_log_probs
        from resmem.fusion import fuse_logits
        from resmem.prompting import build_prompt
        torch, tok = self.torch, self.tokenizer
        encoded = [encode_prompt_answer(tok, prompt=build_prompt(r['question'], prompt_protocol='concise-format-v1'), answer=str(r['answer']), max_length=384) for r in rows]
        batch = collate_encoded(encoded, pad_token_id=int(tok.pad_token_id))
        with torch.inference_mode():
            input_ids = batch.input_ids.to(self.device); attention = batch.attention_mask.to(self.device)
            index, positions = batch.batch_indices.to(self.device), batch.causal_positions.to(self.device)
            joint = self.base(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[index, positions].float()
            if lam != 0.0:
                memory = self.memory(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[index, positions].float()
                joint = fuse_logits(joint, memory, 'memory_decoder', lam)
            nll, _, _ = score_log_probs(torch.nn.functional.log_softmax(joint.float(), dim=-1), batch.target_ids.to(self.device))
            values = nll.cpu().tolist()
        return [[float(x) for x in values[a:b]] for a, b in batch.token_offsets]

def worker(seed, rank, ranks, checkpoint, population_root, base_path, out):
    from resmem.scoring import normalized_metrics
    rt = Runtime(base_path, checkpoint)
    dev = jsonl(Path(population_root) / 'dev.jsonl'); test = jsonl(Path(population_root) / 'test.jsonl')
    dest = Path(out) / f's{seed}'; dest.mkdir(parents=True, exist_ok=False)
    def gen(rows, lam):
        recs = []
        for b in batches(rows):
            for r, p in zip(b, rt.generate(b, lam), strict=True):
                m = normalized_metrics(p['prediction'], r['aliases'])
                recs.append(dict(stable_id=r['stable_id'], dataset=r['dataset'], lambda_=lam, output_ids=p['output_ids'], prediction=p['prediction'], em=m['exact_match'], f1=m['token_f1']))
        return recs
    def nll(rows, lam):
        return {r['stable_id']: s for b in batches(rows) for r, s in zip(b, rt.gold_nll(b, lam), strict=True)}
    def macro(recs, key):
        ds = sorted({r['dataset'] for r in recs}); return float(np.mean([np.mean([r[key] for r in recs if r['dataset'] == d]) for d in ds]))
    dev_scores = {}
    with (dest / 'dev.jsonl').open('w') as h:
        for lam in GRID:
            rr = gen(dev, lam); dev_scores[str(lam)] = dict(em=macro(rr, 'em'), f1=macro(rr, 'f1'))
            for r in rr: h.write(json.dumps(r) + '\n')
    sel = min(GRID, key=lambda l: (-dev_scores[str(l)]['em'], -dev_scores[str(l)]['f1'], l))
    (dest / 'SELECTION.json').write_text(json.dumps(dict(seed=seed, selected=sel, scores=dev_scores, frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=1))
    rr = gen(test, sel); losses = nll(test, sel)
    with (dest / 'test.jsonl').open('w') as h:
        for r in rr: r['token_nll'] = losses[r['stable_id']]; h.write(json.dumps(r) + '\n')
    # Base anchor shard (same full-population batch plan; rank-th of ranks)
    mine = [r for i, b in enumerate(batches(test)) if i % ranks == rank for r in b]
    with (dest / 'base_shard.jsonl').open('w') as h:
        bb = gen(mine, 0.); bn = nll(mine, 0.)
        for r in bb: r['token_nll'] = bn[r['stable_id']]; h.write(json.dumps(r) + '\n')
    (dest / 'COMPLETE.json').write_text(json.dumps(dict(status='complete', seed=seed, selected=sel, checkpoint=str(checkpoint))))

def summarize(out, population_root):
    test = jsonl(Path(population_root) / 'test.jsonl'); ds = sorted({r['dataset'] for r in test})
    def agg(recs):
        by = {d: [r for r in recs if r['dataset'] == d] for d in ds}
        em = np.mean([np.mean([r['em'] for r in by[d]]) for d in ds]); f1 = np.mean([np.mean([r['f1'] for r in by[d]]) for d in ds])
        nll = np.mean([sum(sum(r['token_nll']) for r in by[d]) / sum(len(r['token_nll']) for r in by[d]) for d in ds])
        return dict(em=100 * em, f1=100 * f1, gold_nll=float(nll))
    seeds = {}
    for s in SEEDS:
        d = out / f's{s}'
        if not (d / 'COMPLETE.json').exists(): continue
        seeds[s] = dict(selected=json.loads((d / 'SELECTION.json').read_text())['selected'], **agg(jsonl(d / 'test.jsonl')))
    base = [r for s in SEEDS if (out / f's{s}/base_shard.jsonl').exists() for r in jsonl(out / f's{s}/base_shard.jsonl')]
    res = dict(seeds=seeds, mean={k: float(np.mean([v[k] for v in seeds.values()])) for k in ('em', 'f1', 'gold_nll')} if seeds else None,
               base=agg(base) if len(base) == len(test) else dict(rows=len(base)))
    (out / 'SUMMARY.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

def launch(a):
    from shared_teacher_ablation.eval.lifecycle import run_all
    gpus = a.gpus.split(',') if a.gpus else [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
    out = Path(a.output).absolute(); out.mkdir(parents=True, exist_ok=False); tasks = []
    for rank, s in enumerate(SEEDS):
        argv = [sys.executable, '-m', 'shared_teacher_ablation.memory_decoder.evaluate', '--seed', str(s), '--rank', str(rank), '--checkpoint', a.checkpoint_template.format(seed=s),
                '--population-root', str(Path(a.population_root).absolute()), '--base', str(Path(a.base).absolute()), '--output', str(out)]
        tasks.append(dict(argv=argv, env=dict(os.environ), cwd=str(REPO), log=str(out / f'eval-s{s}.log')))
    try: run_all(tasks, gpus, lambda codes: None)
    finally: summarize(out, a.population_root)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--population-root', required=True); ap.add_argument('--output', type=Path, required=True); ap.add_argument('--base')
    ap.add_argument('--checkpoint-template', help="Memory Decoder checkpoint path with a {seed} field"); ap.add_argument('--gpus')
    ap.add_argument('--seed', type=int); ap.add_argument('--rank', type=int); ap.add_argument('--ranks', type=int, default=len(SEEDS)); ap.add_argument('--checkpoint')
    ap.add_argument('--summarize', action='store_true')
    a = ap.parse_args()
    if a.summarize: summarize(a.output, a.population_root)
    elif a.seed is not None: worker(a.seed, a.rank, a.ranks, a.checkpoint, a.population_root, a.base, a.output)
    else: launch(a)
