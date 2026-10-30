"""Fit on independent QA fit split and rank shared gates on development only.

    python -m gate.search --input runs/gate/fitdev --output runs/gate/search-v1
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from gate.gates import gate_value, feature_vector


def load_pairs(root, allowed=('fit', 'dev')):
    root = Path(root)
    request = json.loads((root/'REQUEST.json').read_text())
    complete = json.loads((root/'COMPLETION.json').read_text())
    assert complete['identity'] == request['identity']
    rows = []
    for p in sorted((root/'tasks').glob('*.json')):
        if p.name.endswith('.receipt.json'): continue
        receipt = json.loads(p.with_suffix('.receipt.json').read_text())
        assert receipt['sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()
        blob = json.loads(p.read_text())
        assert blob['identity'] == request['identity'] == receipt['identity']
        rows.extend(blob['rows'])
    assert len(rows) == complete['rows']
    assert len({r['stable_id'] for r in rows}) == len(rows)
    assert all(r.get('split', 'test') in allowed for r in rows)
    return rows


def decision(row, spec):
    d = row['base']['first_difference']
    return True if d is None else gate_value(d['features'], spec) >= .5


def score(rows, choices):
    out = {}
    for ds in sorted({x['dataset'] for x in rows}):
        indices = [i for i, r in enumerate(rows) if r['dataset'] == ds]
        chosen = [rows[i]['res_scores' if choices[i] else 'base_scores'] for i in indices]
        out[ds] = {'n': len(indices), 'em': float(np.mean([s['exact_match'] for s in chosen]))*100,
                   'f1': float(np.mean([s['token_f1'] for s in chosen]))*100,
                   'avoided_harm': sum(not choices[i] and rows[i]['base_scores']['exact_match'] > rows[i]['res_scores']['exact_match'] for i in indices),
                   'lost_help': sum(not choices[i] and rows[i]['base_scores']['exact_match'] < rows[i]['res_scores']['exact_match'] for i in indices),
                   'gate_off': sum(not choices[i] and rows[i]['base']['first_difference'] is not None for i in indices)}
    return {'macro_em': float(np.mean([x['em'] for x in out.values()])), 'macro_f1': float(np.mean([x['f1'] for x in out.values()])), 'datasets': out}


def threshold(field, value, op='ge'):
    return {'kind': 'threshold', 'field': field, 'threshold': float(value), 'op': op}


def candidates(fit):
    cs = {}
    def add(name, spec, family):
        assert name not in cs
        cs[name] = {'spec': spec, 'mode': 'oneshot', 'family': family}
    add('fixed_res', {'kind': 'constant', 'value': 1}, 'control')
    add('base', {'kind': 'constant', 'value': 0}, 'control')
    entropies = [0, .5, 1, 1.5, 2, 2.5, 3, 4]
    for h in entropies:
        add(f'entropy-{h}', threshold('entropy', h), 'entropy')
        add(f'first-entropy-{h}', threshold('first_entropy', h), 'first_entropy')
    fit_features = [r['base']['first_difference']['features'] for r in fit if r['base']['first_difference']]
    for field in ['pmax', 'margin', 'fused_pmax', 'fused_margin', 'candidate_base_gap', 'residual_advantage', 'overshoot', 'candidate_logrank', 'entropy_change']:
        values = sorted(set(float(x) for x in np.quantile([f[field] for f in fit_features], [.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95])))
        for i, v in enumerate(values):
            for op in ['ge', 'le']:
                add(f'{field}-{op}-{i}', threshold(field, v, op), 'single_confidence')
    for h in entropies:
        for gap in [.1,.25,.5,1,2]:
            for kind in ['all', 'any']:
                add(f'H-{h}-fusedmargin-{gap}-{kind}', {'kind': kind, 'rules': [threshold('entropy', h), threshold('fused_margin', gap)]}, 'entropy_fused_margin')
        for margin in [.25,.5,1,2,4]:
            add(f'H-{h}-margin-{margin}', {'kind': 'all', 'rules': [threshold('entropy', h), threshold('margin', margin, 'le')]}, 'entropy_base_margin')
        for stop_h in entropies + [99]:
            add(f'H-{h}-stop-{stop_h}', {'kind': 'stop_guard', 'threshold': stop_h, 'otherwise': threshold('entropy', h)}, 'stop_guard')
        for later_h in entropies:
            add(f'first-H-{h}-later-H-{later_h}', {'kind': 'position', 'first': threshold('entropy', h), 'later': threshold('entropy', later_h)}, 'position_entropy')
    # Only outcome-discordant fit examples label the route. Others have zero EM utility.
    discordant = [r for r in fit if r['base_scores']['exact_match'] != r['res_scores']['exact_match']]
    assert all(r['base']['first_difference'] is not None for r in discordant)
    y = np.array([r['res_scores']['exact_match'] for r in discordant])
    counts = {ds: sum(r['dataset'] == ds for r in fit) for ds in ['nq', 'triviaqa']}
    weights = np.array([1/counts[r['dataset']] for r in discordant]); weights /= weights.sum()
    fit_details = {'n': len(fit), 'discordant': len(discordant), 'help': int(y.sum()), 'harm': int(len(y)-y.sum()), 'models': []}
    for feature_set in ['basic', 'rich', 'interactions']:
        X = np.array([feature_vector(r['base']['first_difference']['features'], feature_set) for r in discordant])
        mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1
        X = (X-mu)/sd
        for penalty in [.001,.01,.1,1,10]:
            def obj(beta):
                z = X@beta[:-1] + beta[-1]
                loss = np.sum(weights*(np.logaddexp(0,z)-y*z)) + penalty/2*np.sum(beta[:-1]**2)
                error = weights*(expit(z)-y)
                return loss, np.r_[X.T@error + penalty*beta[:-1], error.sum()]
            opt = minimize(obj, np.zeros(X.shape[1]+1), jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            assert opt.success, opt.message
            fit_details['models'].append({'features': feature_set, 'penalty': penalty, 'loss': opt.fun, 'iterations': opt.nit})
            for cut in [.25,.35,.45,.5,.55,.65,.75]:
                spec = {'kind': 'logistic', 'feature_set': feature_set, 'mean': mu.tolist(), 'scale': sd.tolist(), 'coef': opt.x[:-1].tolist(), 'intercept': float(opt.x[-1]), 'threshold': cut, 'penalty': penalty}
                add(f'logistic-{feature_set}-{penalty}-{cut}', spec, 'logistic')
    assert len(cs) < 1000
    return cs, fit_details


def main(args):
    rows = load_pairs(args.input)
    fit = [x for x in rows if x['split'] == 'fit']; dev = [x for x in rows if x['split'] == 'dev']
    cs, fit_details = candidates(fit)
    scores = []
    for name, c in cs.items():
        scores.append({'name': name, 'family': c['family'], **score(dev, [decision(row, c['spec']) for row in dev]), 'complexity': len(json.dumps(c['spec']))})
    scores.sort(key=lambda s: (-s['macro_em'], -s['macro_f1'], s['complexity'], s['name']))
    out = args.output; out.mkdir(exist_ok=True)
    (out/'CANDIDATES.json').write_text(json.dumps(cs, indent=2))
    (out/'LEADERBOARD.json').write_text(json.dumps(scores, indent=2))
    (out/'FIT.json').write_text(json.dumps(fit_details, indent=2))
    best = scores[0]['name']
    (out/'BEST.json').write_text(json.dumps({best: cs[best]}, indent=2))
    families = {}
    for s in scores: families.setdefault(s['family'], s)
    print(json.dumps({'candidates': len(cs), 'fit': fit_details['discordant'], 'best': scores[:12], 'family_bests': families}, indent=2))
    # Bounded actual-rollout development round; learned token policy is explicitly a heuristic transfer.
    tc = {}
    for h in [.5,1,1.5,2]:
        tc[f'token-hard-H-{h}'] = {'mode': 'tokenwise', 'spec': threshold('entropy', h), 'family': 'token_hard_entropy'}
    for h in [.5,1,2]:
        tc[f'token-soft-H-{h}'] = {'mode': 'tokenwise', 'spec': {'kind': 'soft_entropy', 'threshold': h, 'temperature': .5, 'floor': 0}, 'family': 'token_soft_entropy'}
    tc['token-soft-floor'] = {'mode': 'tokenwise', 'spec': {'kind': 'soft_entropy', 'threshold': 1, 'temperature': .5, 'floor': .25}, 'family': 'token_soft_entropy'}
    for family in ['stop_guard', 'entropy_fused_margin', 'logistic']:
        name = families[family]['name']
        tc['token-transfer-'+family] = {**cs[name], 'mode': 'tokenwise', 'family': 'token_transfer_'+family}
    tc['oneshot-winner-parity'] = cs[best]
    (out/'TOKEN_CANDIDATES.json').write_text(json.dumps(tc, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--input', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
