"""Only run after freeze, audit, and final five-set real generation complete.

The test comparator is fixed-strength ResMem and Base from qa.run_grid at the
same per-dataset gamma (``--selected``), e.g.

    python -m qa.run_grid --split-root data/qa/test/frozen --method base --coefficients 0 --output runs/base/test
    python -m qa.run_grid --split-root data/qa/test/frozen --method resmem --checkpoint CKPT \
        --selected runs/resmem/selected.json --output runs/gate/resmem-test
    python -m gate.final_analysis --run-root runs/gate --test-root data/qa/test/frozen \
        --base-grid runs/base/test --resmem-grid runs/gate/resmem-test --selected runs/resmem/selected.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gate.search import load_pairs, score
from gate.rank_rollouts import load_rollouts, summarize, bootstrap, clustered_bootstrap
from qa.common import DATASETS, cell_path, load_rows
from resmem import scoring as scorer

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument('--run-root', type=Path, required=True, help='directory with FROZEN.json, FINAL_CANDIDATE.json, audit-pairs, audit-winner, test-winner')
p.add_argument('--test-root', type=Path, required=True)
p.add_argument('--base-grid', type=Path, required=True)
p.add_argument('--resmem-grid', type=Path, required=True)
p.add_argument('--selected', type=Path, required=True)
args = p.parse_args()
R = args.run_root
freeze = json.loads((R/'FROZEN.json').read_text())
selected = json.loads((R/'FINAL_CANDIDATE.json').read_text())
assert selected == {freeze['candidate_name']: freeze['candidate']}
for p, sha in freeze['inference_code_sha256'].items(): assert hashlib.sha256((HERE/p).read_bytes()).hexdigest() == sha
audit = load_pairs(R/'audit-pairs', allowed=('audit',))
audit_map = {r['stable_id']: r for r in audit}
audit_generated, audit_request = load_rollouts(R/'audit-winner')
test_generated, test_request = load_rollouts(R/'test-winner')
assert audit_request['candidates'] == test_request['candidates'] == selected
for root in ['audit-pairs', 'audit-winner', 'test-winner']:
    completion = json.loads((R/root/'COMPLETION.json').read_text())
    binding = json.loads((R/root/'FREEZE_BINDING.json').read_text())
    assert binding['freeze_sha256'] == hashlib.sha256((R/'FROZEN.json').read_bytes()).hexdigest()
    assert freeze['frozen_at'] <= binding['started_at'] < completion['at']
    request = json.loads((R/root/'REQUEST.json').read_text())
    for key, value in freeze['protocol_identity'].items(): assert request[key] == value
    assert request['code'] == freeze['inference_code_sha256']
for request in [audit_request, test_request]:
    assert request['code'] == freeze['inference_code_sha256']

gammas = json.loads(args.selected.read_text())['coefficients']
assert {ds: float(v) for ds, v in gammas.items()} == test_request['lambdas']
test_map = {}
for ds in DATASETS:
    rows = load_rows(args.test_root/f'{ds}.jsonl')
    base = {r['stable_id']: r for r in load_rows(cell_path(args.base_grid, ds, 0.0, 'predictions'))}
    res = {r['stable_id']: r for r in load_rows(cell_path(args.resmem_grid, ds, float(gammas[ds]), 'predictions'))}
    for row in rows:
        b, a = base[row['stable_id']], res[row['stable_id']]
        bs, rs = scorer.normalized_metrics(b['prediction'], row['aliases']), scorer.normalized_metrics(a['prediction'], row['aliases'])
        bid, aid = b['output_ids'], a['output_ids']
        k = next((i for i, (x, y) in enumerate(zip(bid+[None], aid+[None])) if x != y), None)
        test_map[row['stable_id']] = {**row, 'base_scores': bs, 'res_scores': rs,
            'base': {'first_difference': None if k is None else {'position': k+1}, 'prediction': b['prediction']}, 'res': {'prediction': a['prediction']}}
summary = {'frozen': freeze}
details = []
for split, candidates, baselines in [('audit', audit_generated, audit_map), ('test', test_generated, test_map)]:
    rows = candidates[freeze['candidate_name']]
    assert {x['stable_id'] for x in rows} == set(baselines)
    for row in rows:
        assert row['question'] == baselines[row['stable_id']]['question']
        fresh = scorer.normalized_metrics(row['prediction']['prediction'], row['aliases'])
        assert all(abs(fresh[k]-row['scores'][k]) < 1e-12 for k in fresh)
    originals = list(baselines.values())
    entry = {'base': score(originals, [False]*len(originals)), 'fixed_res': score(originals, [True]*len(originals)),
             'winner': summarize(rows, baselines), 'paired_em_bootstrap': (bootstrap(rows, baselines) if split == 'audit' else clustered_bootstrap(rows, baselines))}
    mechanisms = {'prevented_base_correct_res_wrong': 0, 'lost_res_corrections': 0,
                  'both_wrong_gate_correct': 0, 'both_correct_gate_wrong': 0}
    intervention_entropy = []; intervention_positions = []; blocked = 0; applied_weights = []
    for row in rows:
        old = baselines[row['stable_id']]
        b = old['base_scores']['exact_match']; r = old['res_scores']['exact_match']; g = row['scores']['exact_match']
        mechanisms['prevented_base_correct_res_wrong'] += int(b == 1 and r == 0 and g == 1)
        mechanisms['lost_res_corrections'] += int(b == 0 and r == 1 and g == 0)
        mechanisms['both_wrong_gate_correct'] += int(b == 0 and r == 0 and g == 1)
        mechanisms['both_correct_gate_wrong'] += int(b == 1 and r == 1 and g == 0)
        for position, e in enumerate(row['prediction']['gate_events'], 1):
            blocked += int(e['disagreement'] and not e['intervened'])
            if e['intervened']:
                intervention_entropy.append(e['entropy']); intervention_positions.append(position); applied_weights.append(e['weight'])
    entry['mechanisms'] = mechanisms
    assert sum(x['gains_vs_res'] for x in entry['winner']['datasets'].values()) == mechanisms['prevented_base_correct_res_wrong']+mechanisms['both_wrong_gate_correct']
    assert sum(x['losses_vs_res'] for x in entry['winner']['datasets'].values()) == mechanisms['lost_res_corrections']+mechanisms['both_correct_gate_wrong']
    entry['intervention_timing'] = {'first_token_interventions': sum(x == 1 for x in intervention_positions),
        'later_token_interventions': sum(x > 1 for x in intervention_positions), 'blocked_full_res_disagreements': blocked,
        'entropy_q10_q50_q90_nats': np.quantile(intervention_entropy, [.1,.5,.9]).tolist() if intervention_entropy else [],
        'weight_q10_q50_q90': np.quantile(applied_weights, [.1,.5,.9]).tolist() if applied_weights else []}
    summary[split] = entry
    for row in rows:
        b = baselines[row['stable_id']]; new = row['scores']['exact_match']; old_em = b['res_scores']['exact_match']
        if new != old_em:
            details.append({'split': split, 'dataset': row['dataset'], 'stable_id': row['stable_id'], 'question': row['question'],
                            'aliases': row['aliases'], 'base': b['base']['prediction'], 'fixed_res': b['res']['prediction'],
                            'gate': row['prediction']['prediction'], 'effect': 'gain' if new > old_em else 'loss',
                            'first_difference': row['prediction']['first_difference'], 'gate_events': row['prediction']['gate_events']})
summary['supported_replacement'] = all(summary[s]['winner']['macro_em'] > summary[s]['fixed_res']['macro_em'] for s in ['audit','test'])
summary['all_test_domains_improved'] = all(summary['test']['winner']['datasets'][ds]['em'] > summary['test']['fixed_res']['datasets'][ds]['em'] for ds in summary['test']['winner']['datasets'])
summary['fresh_scoring_verified'] = True
(R/'FINAL_SUMMARY.json').write_text(json.dumps(summary, indent=2))
(R/'FINAL_ERROR_TRANSITIONS.json').write_text(json.dumps(details, indent=2))
print(json.dumps({k: v for k,v in summary.items() if k != 'frozen'}, indent=2))
