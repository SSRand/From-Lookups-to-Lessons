"""Validate complete actual generations and rank them against fixed RES.

    python -m gate.rank_rollouts --pairs runs/gate/fitdev --search runs/gate/search-round3 \
        --rollouts runs/gate/token-v1 runs/gate/token-round3 runs/gate/token-round3-soft --output runs/gate/development-final
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from gate.search import load_pairs, decision


def load_rollouts(root):
    root = Path(root); request = json.loads((root/'REQUEST.json').read_text())
    complete = json.loads((root/'COMPLETION.json').read_text())
    assert complete['identity'] == request['identity']
    result = {k: [] for k in request['candidates']}
    tasks = 0
    for path in sorted((root/'tasks').glob('*.json')):
        if path.name.endswith('.receipt.json'): continue
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        blob = json.loads(path.read_text()); assert blob['identity'] == receipt['identity'] == request['identity']
        result[blob['job']['candidate']].extend(blob['rows']); tasks += 1
    assert tasks == complete['jobs'] and sum(len(x) for x in result.values()) == complete['rows']
    for rows in result.values(): assert len({x['stable_id'] for x in rows}) == len(rows)
    return result, request


def summarize(rows, baselines):
    out = {}; all_events = []
    for ds in sorted({x['dataset'] for x in rows}):
        group = [x for x in rows if x['dataset'] == ds]
        ref = [baselines[x['stable_id']] for x in group]
        em = np.array([x['scores']['exact_match'] for x in group]); f1 = np.array([x['scores']['token_f1'] for x in group])
        rem = np.array([x['res_scores']['exact_match'] for x in ref])
        bem = np.array([x['base_scores']['exact_match'] for x in ref])
        out[ds] = {'n': len(group), 'em': float(em.mean()*100), 'f1': float(f1.mean()*100),
                   'gains_vs_res': int((em > rem).sum()), 'losses_vs_res': int((em < rem).sum()),
                   'base_correct_now_wrong': int(((bem == 1) & (em == 0)).sum()),
                   'base_wrong_now_correct': int(((bem == 0) & (em == 1)).sum())}
        domain_events = [e for x in group for e in x['prediction']['gate_events']]
        proposed = sum(e['disagreement'] for e in domain_events)
        intervened = sum(e.get('intervened', False) for e in domain_events)
        out[ds]['proposed_disagreements'] = proposed
        out[ds]['actual_interventions'] = intervened
        out[ds]['proposal_acceptance_rate'] = intervened/proposed if proposed else 0
        all_events.extend(domain_events)
    events = {'active_steps': len(all_events), 'disagreement_steps': sum(e['disagreement'] for e in all_events),
              'actual_interventions': sum(e.get('intervened', False) for e in all_events),
              'third_token_steps': sum(e.get('third_token', False) for e in all_events),
              'nonempty_stop_interventions': sum(e.get('intervened', False) and e['stop'] for e in all_events)}
    return {'macro_em': float(np.mean([v['em'] for v in out.values()])), 'macro_f1': float(np.mean([v['f1'] for v in out.values()])), 'datasets': out, 'events': events}


def bootstrap(rows, baselines, repetitions=10000):
    rng = np.random.default_rng(20260920)
    macro = np.zeros(repetitions); result = {}
    domains = sorted({x['dataset'] for x in rows})
    for ds in domains:
        group = [x for x in rows if x['dataset'] == ds]
        deltas = np.array([x['scores']['exact_match']-baselines[x['stable_id']]['res_scores']['exact_match'] for x in group])
        # Equivalent grouped multinomial bootstrap of ternary paired outcomes, avoids giant matrices.
        probs = np.array([(deltas == -1).mean(), (deltas == 0).mean(), (deltas == 1).mean()])
        counts = rng.multinomial(len(group), probs, size=repetitions)
        b = (counts[:, 2]-counts[:, 0])/len(group)*100
        macro += b/len(domains)
        result[ds] = {'delta_em_pp': float(deltas.mean()*100), 'ci95': np.quantile(b, [.025,.975]).tolist()}
    result['macro'] = {'delta_em_pp': float(np.mean([r['delta_em_pp'] for r in result.values()])), 'ci95': np.quantile(macro, [.025,.975]).tolist()}
    return result


def clustered_bootstrap(rows, baselines, repetitions=10000):
    """Resample normalized question clusters jointly across test datasets.

    Collapse clusters with the same sufficient statistics into multinomial bins;
    this is exactly the same EM bootstrap distribution with much less memory.
    """
    from qa.data.build_dpr_dev import norm
    domains = sorted({x['dataset'] for x in rows}); d = len(domains)
    groups = {}
    for row in rows:
        key = norm(row['question']); ds = domains.index(row['dataset'])
        v = groups.setdefault(key, [0]*(2*d))
        v[ds] += 1
        v[d+ds] += int(row['scores']['exact_match']-baselines[row['stable_id']]['res_scores']['exact_match'])
    bins = collections.Counter(tuple(v) for v in groups.values())
    vectors = np.array(list(bins)); counts = np.array(list(bins.values()))
    rng = np.random.default_rng(20260920)
    draws = rng.multinomial(len(groups), counts/counts.sum(), size=repetitions)
    denominators = draws@vectors[:, :d]; numerators = draws@vectors[:, d:]
    assert np.all(denominators > 0)
    boot = numerators/denominators*100
    point = (counts@vectors[:, d:])/(counts@vectors[:, :d])*100
    result = {ds: {'delta_em_pp': float(point[i]), 'ci95': np.quantile(boot[:, i], [.025,.975]).tolist()} for i, ds in enumerate(domains)}
    result['macro'] = {'delta_em_pp': float(point.mean()), 'ci95': np.quantile(boot.mean(1), [.025,.975]).tolist()}
    result['method'] = {'kind': 'normalized-question-cluster bootstrap', 'clusters': len(groups), 'rows': len(rows), 'sufficient_statistic_bins': len(bins), 'replicates': repetitions}
    return result


def main(args):
    pairs = [r for r in load_pairs(args.pairs) if r['split'] == 'dev']; bm = {r['stable_id']: r for r in pairs}
    parent = json.loads((args.pairs/'REQUEST.json').read_text())
    ranking = json.loads((args.search/'LEADERBOARD.json').read_text())
    cs = json.loads((args.search/'CANDIDATES.json').read_text())
    parity = {}; actual = {}
    for root in args.rollouts:
        predictions, request = load_rollouts(root)
        assert request['splits'] == 'dev' and request['scope'] == 'calibration'
        for key in ['data_sha', 'lambdas', 'protocol', 'batch_size']:
            assert request[key] == parent[key], ('protocol drift', key)
        for name, rows in predictions.items():
            assert {x['stable_id'] for x in rows} == set(bm)
            c = request['candidates'][name]
            if c['mode'] == 'oneshot':
                mismatches = []
                for row in rows:
                    old = bm[row['stable_id']]
                    expected = old['res' if decision(old, c['spec']) else 'base']
                    if expected['output_ids'] != row['prediction']['output_ids']: mismatches.append(row['stable_id'])
                parity[name] = {'rows': len(rows), 'mismatches': mismatches}
                assert not mismatches, parity[name]
            actual[name] = {'name': name, 'family': c['family'], **summarize(rows, bm), 'complexity': len(json.dumps(c['spec']))}
            if name in cs:
                assert cs[name] == c, ('candidate name reused', name)
                old = next(s for s in ranking if s['name'] == name)
                assert abs(old['macro_em']-actual[name]['macro_em']) < 1e-10 and abs(old['macro_f1']-actual[name]['macro_f1']) < 1e-10
            else:
                cs[name] = c; ranking.append(actual[name])
    ranking.sort(key=lambda s: (-s['macro_em'], -s['macro_f1'], s['complexity'], s['name']))
    args.output.mkdir(exist_ok=True)
    for name, obj in [('LEADERBOARD.json', ranking), ('CANDIDATES.json', cs), ('ACTUAL.json', actual), ('PARITY.json', parity)]:
        (args.output/name).write_text(json.dumps(obj, indent=2))
    print(json.dumps({'best': ranking[:12], 'actual': actual, 'parity': parity}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--pairs', type=Path, required=True); p.add_argument('--search', type=Path, required=True)
    p.add_argument('--rollouts', type=Path, nargs='+', required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
