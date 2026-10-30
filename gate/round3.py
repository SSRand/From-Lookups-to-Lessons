"""Final development iteration: narrowly guard year-to-ISO-date extensions.

The ``year_dash`` feature is recorded causally by gate.runner at the first
Base/ResMem disagreement.

    python -m gate.round3 --pairs runs/gate/fitdev --search runs/gate/search-v1 --output runs/gate/search-round3
"""
import argparse
import json
from pathlib import Path
from gate.search import load_pairs, score, decision, threshold

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument('--pairs', type=Path, required=True); p.add_argument('--search', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
rows = load_pairs(args.pairs)
cs = json.loads((args.search/'CANDIDATES.json').read_text())
lb = json.loads((args.search/'LEADERBOARD.json').read_text())
family_best = {}
for x in lb: family_best.setdefault(x['family'], x['name'])
new = {}; dev = [x for x in rows if x['split'] == 'dev']
for family in ['control', 'logistic', 'entropy_fused_margin', 'stop_guard']:
    parent = 'fixed_res' if family == 'control' else family_best[family]
    name = 'format-guard-'+family
    spec = {'kind': 'all', 'rules': [threshold('year_dash', .5, 'le'), cs[parent]['spec']]}
    new[name] = {'mode': 'oneshot', 'spec': spec, 'family': 'format_guard_'+family}
    lb.append({'name': name, 'family': new[name]['family'], **score(dev, [decision(row, spec) for row in dev]), 'complexity': len(json.dumps(spec))})
cs.update(new)
lb.sort(key=lambda s: (-s['macro_em'], -s['macro_f1'], s['complexity'], s['name']))
out = args.output; out.mkdir(exist_ok=True)
(out/'CANDIDATES.json').write_text(json.dumps(cs, indent=2))
(out/'LEADERBOARD.json').write_text(json.dumps(lb, indent=2))
rollouts = {'token-'+k: {**v, 'mode': 'tokenwise'} for k, v in new.items()}
# Verify best third-round once-route with real generation, even if the old incumbent is stronger.
best_new = next(x['name'] for x in lb if x['name'] in new)
rollouts[best_new] = new[best_new]
rollouts['hybrid-logistic-year'] = {'mode': 'hybrid', 'family': 'hybrid', 'spec': {'kind': 'hybrid',
    'route': cs[family_best['logistic']]['spec'], 'guard': threshold('year_dash', .5, 'le')}}
rollouts['hybrid-margin-stop'] = {'mode': 'hybrid', 'family': 'hybrid', 'spec': {'kind': 'hybrid',
    'route': cs[family_best['entropy_fused_margin']]['spec'], 'guard': cs[family_best['stop_guard']]['spec']}}
(out/'TOKEN_ROUND3.json').write_text(json.dumps(rollouts, indent=2))
print(json.dumps({'new_results': [x for x in lb if x['name'] in new], 'overall_best': lb[0]}, indent=2))
