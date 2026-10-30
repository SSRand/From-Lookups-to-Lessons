"""Tokenwise-gated ResMem generation: qa.runtime decoding with an explicit causal fusion policy.

The policy replaces the fusion step of ``QARuntime.generate``; all decoding
(384 input tokens, 12 new tokens, full-prefix greedy, batch 16) is unchanged.
ResMem uses the per-dataset gamma of ``--selected``. Work is split into
256-question jobs; run one process per GPU with ``--shard i --num-shards n``.

    # base/full-ResMem pairs with first-difference features (gate fitting and selection)
    python -m gate.runner --phase collect --scope calibration --splits fit,dev \
        --calibration data/qa/dpr_dev/calibration.jsonl --checkpoint CKPT --selected runs/resmem/selected.json \
        --output runs/gate/fitdev
    # real rollouts of candidate policies
    python -m gate.runner --phase evaluate --scope calibration --splits dev --candidates runs/gate/search-v1/TOKEN_CANDIDATES.json ...
    python -m gate.runner --phase evaluate --scope test --splits test --test-root data/qa/test/frozen \
        --candidates gate/FINAL_CANDIDATE.json ...
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch

from gate.gates import gate_value, validate_spec
from qa.common import DATASETS, load_rows
from resmem.fusion import RESIDUAL, fuse_logits
from resmem.scoring import normalized_metrics

HERE = Path(__file__).resolve().parent
PROTOCOL = {'batch_size': 16, 'max_input_length': 384, 'max_new_tokens': 12, 'prompt_protocol': 'concise-format-v1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class FusionPolicy:
    def __init__(self, rt, rows, lam, spec, mode):
        validate_spec(spec)
        assert mode in ['oneshot', 'tokenwise', 'hybrid']
        assert (spec['kind'] == 'hybrid') == (mode == 'hybrid')
        self.rt, self.rows, self.lam, self.spec, self.mode = rt, rows, lam, spec, mode
        n = len(rows)
        self.prefix = [[] for _ in rows]; self.done = [False]*n
        self.first = [None]*n; self.diff = [None]*n; self.route = [None]*n
        self.step = 0; self.events = []
        self.text_cache = {}

    def token_text(self, token):
        if token not in self.text_cache:
            self.text_cache[token] = self.rt.tokenizer.decode([token], skip_special_tokens=False)
        return self.text_cache[token]

    def __call__(self, base, memory, coefficient):
        t = torch
        fused = fuse_logits(base, memory, RESIDUAL, coefficient)
        b = base.argmax(-1); a = fused.argmax(-1)
        lp = t.log_softmax(base, -1); fp = t.log_softmax(fused, -1)
        top = base.topk(2, -1).values; ft = fused.topk(2, -1).values
        bi = b[:, None]; ai = a[:, None]
        gap = (base.gather(1, bi)-base.gather(1, ai)).squeeze(1)
        advantage = self.lam*(memory.gather(1, ai)-memory.gather(1, bi)).squeeze(1)
        values = {
            'entropy': -(lp.exp()*lp).sum(-1), 'pmax': lp.max(-1).values.exp(), 'margin': top[:, 0]-top[:, 1],
            'fused_entropy': -(fp.exp()*fp).sum(-1), 'fused_pmax': fp.max(-1).values.exp(), 'fused_margin': ft[:, 0]-ft[:, 1],
            'candidate_base_gap': gap, 'residual_advantage': advantage, 'overshoot': (advantage-gap)/(gap+.25),
            'candidate_logrank': ((base > base.gather(1, ai)).sum(-1).float()+1).log(),
            'residual_std': memory.std(-1)*self.lam,
        }
        cpu = {k: v.detach().cpu().tolist() for k, v in values.items()}
        bs, rs = b.tolist(), a.tolist(); self.step += 1
        weights = []; step_events = []
        for i in range(len(self.rows)):
            prefix = self.rt.tokenizer.decode(self.prefix[i], skip_special_tokens=True)
            f = {k: v[i] for k, v in cpu.items()}
            f.update(entropy_change=f['fused_entropy']-f['entropy'], **{'lambda': self.lam}, position=self.step,
                     first_entropy=f['entropy'] if self.first[i] is None else self.first[i]['entropy'],
                     base_stop=float(bs[i] == self.rt.tokenizer.eos_token_id or '\n' in self.token_text(bs[i])),
                     res_stop=float(rs[i] == self.rt.tokenizer.eos_token_id or '\n' in self.token_text(rs[i])),
                     prefix_nonempty=float(bool(prefix.strip())), prefix_words=len(prefix.split()),
                     prefix_chars=len(prefix.strip()), prefix_numeric=float(any(c.isdigit() for c in prefix)),
                     question_words=len(self.rows[i]['question'].split()))
            f['year_dash'] = float(bool(f['base_stop'] and re.fullmatch(r'\d{4}', prefix.strip()) and self.token_text(rs[i]).strip().startswith('-')))
            if not self.done[i] and self.first[i] is None:
                self.first[i] = f
            if not self.done[i] and bs[i] != rs[i] and self.diff[i] is None:
                self.diff[i] = {'features': f, 'position': self.step, 'base_token': bs[i], 'res_token': rs[i], 'prefix_ids': list(self.prefix[i])}
            w = gate_value(f, self.spec['route'] if self.mode == 'hybrid' else self.spec)
            if self.mode in ['oneshot', 'hybrid']:
                if self.route[i] is None and bs[i] != rs[i] and not self.done[i]:
                    self.route[i] = float(w >= .5)
                w = 1.0 if self.route[i] is None else self.route[i]
                if self.mode == 'hybrid':
                    w *= gate_value(f, self.spec['guard'])
            weights.append(w)
            step_events.append({'active': not self.done[i], 'disagreement': bs[i] != rs[i], 'weight': w, 'stop': bool(f['base_stop'] and f['prefix_nonempty'])})
        # Select exact endpoints rather than recomputing the full residual at 1.
        w = t.tensor(weights, device=base.device, dtype=t.float32)[:, None]
        joint = t.where(w == 1, fused, t.where(w == 0, base, base + self.lam*w*memory))
        chosen = joint.argmax(-1).tolist()
        for i, token in enumerate(chosen):
            self_event = step_events[i]
            self_event.update(base_token=bs[i], res_token=rs[i], chosen_token=token,
                              intervened=token != bs[i], third_token=token not in [bs[i], rs[i]],
                              entropy=cpu['entropy'][i])
            if not self.done[i]:
                if token == self.rt.tokenizer.eos_token_id:
                    self.done[i] = True
                else:
                    self.prefix[i].append(token)
        self.events.append(step_events)
        return joint

    def generate(self):
        self.rt._fuse = self
        try:
            # All actual decoding remains in qa.runtime.QARuntime.generate.
            predictions = self.rt.generate(self.rows, self.lam, 16)
        finally:
            del self.rt._fuse
        for i, row in enumerate(predictions):
            assert row['output_ids'] == self.prefix[i]
            row.update(first_features=self.first[i], first_difference=self.diff[i], gate_events=[s[i] for s in self.events if s[i]['active']])
        return predictions


def generate(rt, rows, lam, spec, mode='tokenwise'):
    return FusionPolicy(rt, rows, lam, spec, mode).generate()


def run_job(rt, job, request, lambdas):
    started = time.time(); results = []
    for start in range(0, len(job['rows']), 16):
        rows = job['rows'][start:start+16]; lam = lambdas[job['dataset']]
        if request['phase'] in ['collect', 'canary']:
            bp = generate(rt, rows, lam, {'kind': 'constant', 'value': 0})
            rp = generate(rt, rows, lam, {'kind': 'constant', 'value': 1})
            if request['phase'] == 'canary':
                assert [x['output_ids'] for x in rt.generate(rows, 0, 16)] == [x['output_ids'] for x in bp]
                assert [x['output_ids'] for x in rt.generate(rows, lam, 16)] == [x['output_ids'] for x in rp]
            for src, b, a in zip(rows, bp, rp):
                diff = b['first_difference']; eos = rt.tokenizer.eos_token_id
                if diff:
                    k = diff['position']-1; aid = a['output_ids']+[eos]; bid = b['output_ids']+[eos]
                    assert bid[:k] == aid[:k] == diff['prefix_ids'] and aid[k] == diff['res_token'] and bid[k] == diff['base_token']
                else:
                    assert b['output_ids'] == a['output_ids']
                results.append({**src, 'base': b, 'res': a,
                    'base_scores': normalized_metrics(b['prediction'], src['aliases']),
                    'res_scores': normalized_metrics(a['prediction'], src['aliases'])})
        else:
            candidate = request['candidates'][job['candidate']]
            pp = generate(rt, rows, lam, candidate['spec'], candidate['mode'])
            results.extend({**src, 'prediction': p, 'scores': normalized_metrics(p['prediction'], src['aliases'])} for src, p in zip(rows, pp))
    return results, time.time()-started


def main(args):
    lambdas = {ds: float(v) for ds, v in json.loads(args.selected.read_text())['coefficients'].items()}
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(args.candidates.read_text()) if args.candidates else {}
    request = {'phase': args.phase, 'scope': args.scope,
               'code': {p.name: sha(p) for p in [Path(__file__), HERE/'gates.py']},
               'data_sha': sha(args.calibration), 'candidates': candidates, 'splits': args.splits,
               'protocol': PROTOCOL, 'lambdas': lambdas, 'batch_size': 16}
    request['identity'] = hashlib.sha256(canonical(request)).hexdigest()
    if (root/'REQUEST.json').exists(): assert json.loads((root/'REQUEST.json').read_text()) == request
    else: atomic(root/'REQUEST.json', request)
    if args.scope == 'calibration':
        source = load_rows(args.calibration)
        source = [x for x in source if x['split'] in args.splits.split(',')]
        data = {f'{ds}-{split}': [x for x in source if x['dataset'] == ds and x['split'] == split]
                for ds in ['nq', 'triviaqa'] for split in args.splits.split(',')}
    else: data = {ds: load_rows(args.test_root/f'{ds}.jsonl') for ds in DATASETS}
    if args.phase == 'canary': data = {ds: rows[:32] for ds, rows in data.items()}
    jobs = []
    for candidate in list(candidates) if candidates else ['pairs']:
        for group, rows in data.items():
            for start in range(0, len(rows), 256):
                jobs.append({'id': f'{candidate}-{group}-{start:05d}', 'candidate': candidate, 'dataset': rows[0]['dataset'], 'rows': rows[start:start+256]})
    pending = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard
               and not (root/'tasks'/f"{job['id']}.receipt.json").exists()]
    if pending:
        from qa.runtime import QARuntime
        rt = QARuntime(args.base_model)
        rt.set_method(RESIDUAL, args.checkpoint)
        for job in pending:
            results, seconds = run_job(rt, job, request, lambdas)
            payload = {'job': {k: v for k, v in job.items() if k != 'rows'}, 'identity': request['identity'], 'elapsed_seconds': seconds, 'rows': results}
            path = root/'tasks'/f"{job['id']}.json"
            atomic(path, payload)
            atomic(path.with_suffix('.receipt.json'), {'sha256': sha(path), 'rows': len(results), 'identity': request['identity']})
            print(json.dumps({'job': job['id'], 'rows': len(results), 'seconds': seconds}), flush=True)
    for job in jobs:
        path = root/'tasks'/f"{job['id']}.json"
        if not path.with_suffix('.receipt.json').exists(): return
    for job in jobs:
        path = root/'tasks'/f"{job['id']}.json"
        meta = json.loads(path.with_suffix('.receipt.json').read_text())
        blob = json.loads(path.read_text())
        assert meta['identity'] == blob['identity'] == request['identity'] and meta['sha256'] == sha(path)
        assert meta['rows'] == len(job['rows']) == len(blob['rows'])
        assert [x['stable_id'] for x in blob['rows']] == [x['stable_id'] for x in job['rows']]
    atomic(root/'COMPLETION.json', {'status': 'complete', 'identity': request['identity'], 'jobs': len(jobs), 'rows': sum(len(j['rows']) for j in jobs), 'at': time.time()})


def parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--phase', choices=['collect', 'canary', 'evaluate'], required=True)
    p.add_argument('--scope', choices=['calibration', 'test'], default='calibration')
    p.add_argument('--splits', default='fit,dev')
    p.add_argument('--candidates', type=Path)
    p.add_argument('--calibration', type=Path, required=True, help='calibration.jsonl from qa.data.build_dpr_dev')
    p.add_argument('--test-root', type=Path, help='frozen test <dataset>.jsonl files (scope test)')
    p.add_argument('--selected', type=Path, required=True, help='per-dataset ResMem gamma (selected.json format)')
    p.add_argument('--checkpoint', required=True, help='ResMem memory checkpoint')
    p.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
