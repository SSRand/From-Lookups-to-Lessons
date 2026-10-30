"""Memory Decoder (standalone, mixture readout) vs the ablation's native MLP (standalone+mixture) and RES (joint+additive).
Primary metric as in the ablation: equal-dataset macro, averaged over 8 paired seeds, all 38,627 rows.
CI: 10,000-draw question-cluster bootstrap (NFKC question_group), stratified by dataset membership,
shared weights across methods and seeds (conditional on trained models).

  python -m shared_teacher_ablation.memory_decoder.aggregate AGG   (directory built by stage_inputs)
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); SEEDS = (42, 123, 456, 789, 2026, 2027, 2028, 2029)
def jl(p): return [json.loads(x) for x in open(p)]
test = jl(D / 'test.jsonl'); ids = [r['stable_id'] for r in test]; pos = {s: i for i, s in enumerate(ids)}; N = len(test)
ds = sorted({r['dataset'] for r in test}); dsi = np.array([ds.index(r['dataset']) for r in test])
def load(path):
    a = np.full((N, 3), np.nan)  # em, f1, nll_sum ; plus token count
    cnt = np.zeros(N)
    for r in jl(path):
        i = pos[r['stable_id']]; a[i] = (r['em'], r['f1'], sum(r['token_nll'])); cnt[i] = len(r['token_nll'])
    assert not np.isnan(a).any(); return a, cnt
arms = {'md': {s: load(D / 'md' / f's{s}.jsonl') for s in SEEDS},
        'mlp': {s: load(D / 'ctrl' / f'standalone_memory-s{s}.jsonl') for s in SEEDS},
        'res': {s: load(D / 'ctrl' / f'joint_base_anchored-s{s}.jsonl') for s in SEEDS}}
base_rows = [r for s in SEEDS for r in jl(D / 'md' / f'base{s}.jsonl')]
assert len(base_rows) == N
tmp = D / 'base_all.jsonl'; tmp.write_text(''.join(json.dumps(r) + '\n' for r in base_rows)); base = load(tmp)
grp = {}; gi = np.array([grp.setdefault(r['question_group'], len(grp)) for r in test]); G = len(grp)
memb = defaultdict(set)
for g, d in zip(gi, dsi): memb[g].add(d)
strata = defaultdict(list)
for g in range(G): strata[tuple(sorted(memb[g]))].append(g)
def macro(a, cnt, w=None):
    w = np.ones(N) if w is None else w
    em = np.mean([(w * a[:, 0])[dsi == k].sum() / w[dsi == k].sum() for k in range(len(ds))])
    f1 = np.mean([(w * a[:, 1])[dsi == k].sum() / w[dsi == k].sum() for k in range(len(ds))])
    nll = np.mean([(w * a[:, 2])[dsi == k].sum() / (w * cnt)[dsi == k].sum() for k in range(len(ds))])
    return np.array([100 * em, 100 * f1, nll])
point = {m: {s: macro(*v) for s, v in arms[m].items()} for m in arms}
mean = {m: np.mean(list(point[m].values()), 0) for m in arms}
bpt = macro(*base)
# bootstrap on seed-averaged per-question arrays (macro is linear in per-row values given weights, so average first)
avg = {m: (np.mean([arms[m][s][0] for s in SEEDS], 0), np.mean([arms[m][s][1] for s in SEEDS], 0)) for m in arms}
rng = np.random.default_rng(20260921); R = 10000
# cluster-level sums per dataset for speed
def cl_sums(v):
    out = np.zeros((len(ds), G)); np.add.at(out, (dsi, gi), v); return out
cnt_rows = cl_sums(np.ones(N))
S = {m: [cl_sums(avg[m][0][:, j]) for j in range(3)] + [cl_sums(avg[m][1])] for m in list(arms)}
S['base'] = [cl_sums(base[0][:, j]) for j in range(3)] + [cl_sums(base[1])]
draws = {k: [] for k in ('md-mlp', 'md-res', 'res-mlp', 'md-base', 'mlp-base', 'res-base')}
for start in range(0, R, 250):
    n = min(250, R - start); W = np.zeros((n, G))
    for idx in strata.values():
        idx = np.array(idx); W[:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n)
    den = W @ cnt_rows.T  # n x ds
    val = {}
    for m, (e, f, l, c) in S.items():
        val[m] = np.stack([100 * (W @ e.T / den).mean(1), 100 * (W @ f.T / den).mean(1), ((W @ l.T) / (W @ c.T)).mean(1)], 1)
    for k in draws:
        a, b = k.split('-'); draws[k].append(val[a] - val[b])
ci = {k: np.quantile(np.concatenate(v), [.025, .975], axis=0).T.tolist() for k, v in draws.items()}
est = {'md-mlp': mean['md'] - mean['mlp'], 'md-res': mean['md'] - mean['res'], 'res-mlp': mean['res'] - mean['mlp'],
       'md-base': mean['md'] - bpt, 'mlp-base': mean['mlp'] - bpt, 'res-base': mean['res'] - bpt}
per_seed = {s: {m: point[m][s].round(4).tolist() for m in arms} for s in SEEDS}
wins = {k: int(sum(point[a][s][1] > point[b][s][1] for s in SEEDS)) for k, (a, b) in {'md>mlp': ('md', 'mlp'), 'md>res': ('md', 'res')}.items()}
out = dict(metric_order=['EM', 'F1', 'gold_NLL'], base=bpt.tolist(), mean={m: v.tolist() for m, v in mean.items()},
           contrasts={k: dict(estimate=est[k].tolist(), ci95=ci[k]) for k in est}, per_seed=per_seed, seed_f1_wins=wins,
           bootstrap=dict(repeats=R, seed=20260921, unit='question_group (37,419)', strata='dataset membership', pairing='shared weights; seed-averaged per-question values'))
(D / 'MD_AGGREGATE.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ('base', 'mean', 'contrasts', 'seed_f1_wins')}, indent=1)); print(json.dumps(per_seed))
