"""Local CPU seal of all 32 dev policies before any test evaluation.

  python -m shared_teacher_ablation.eval.barrier --cal-root EVAL/calibration [--cal-root ...] --population-root POP --output SEAL.json
"""
import argparse
from pathlib import Path
from . import protocol as p
from . import population,outputs,selection

def open_seal(path,expected=None,c=None):
 if expected is not None and p.sha(path)!=expected:raise ValueError('global seal SHA')
 b=p.read(path)
 if b['schema']!='tec-global-dev-seal-v1' or b['complete'] is not True or b['source_manifest_sha256']!=p.source():raise ValueError('global seal identity')
 seeds=[s for v in b['cohorts'].values() for s in v['seeds']]
 if sorted(seeds)!=sorted(p.SEEDS):raise ValueError('seal must cover every registered seed once')
 selection.validate_policy_keys(b['policies'])
 if c is not None:
  host=b['cohorts'][c['cohort']]
  if host['config_sha256']!=c['_sha256'] or host['training_complete_sha256']!=c['parent']['training_complete_sha256'] or b['population_sha256']!=c['population']['COMPLETE.json']['sha256']:raise ValueError('seal cohort/config/data binding')
  for cid,m in c['models'].items():
   for r in p.READOUTS:
    v=b['policies'][p.key(m['seed'],m['objective'],r)]
    if v['checkpoint_sha256']!=m['files']['model.safetensors']['sha256']:raise ValueError('seal checkpoint binding')
 return b

def seal(cal_roots,pop_root,pop_sha,destination):
 rr,done=population.load_bundle(pop_root,pop_sha);policies={};hosts={};array_identity=None;inits={};base_zero=None;train_source=None
 for directory in cal_roots:
  root=Path(directory);receipt=p.read(root/'CALIBRATION_COMPLETE.json');c=p.read(root/'CONFIG.json');c['_sha256']=p.sha(root/'CONFIG.json');cohort=c['cohort']
  if train_source is None:train_source=c['parent']['source_manifest_sha256']
  if c['source_manifest_sha256']!=p.source() or c['parent']['source_manifest_sha256']!=train_source:raise ValueError('cal config/producer source')
  if cohort in hosts or receipt['status']!='complete' or receipt['complete'] is not True or receipt['config_sha256']!=c['_sha256'] or receipt['source_manifest_sha256']!=p.source() or c['population']['COMPLETE.json']['sha256']!=pop_sha:raise ValueError('actual complete cohort calibration')
  if type(receipt['policies'])is not int or receipt['policies']!=2*len(c['models']) or type(receipt['completed_models'])is not int or receipt['completed_models']!=len(c['models']):raise ValueError('calibration full count')
  for rel,h in receipt['artifacts'].items():
   q=p.physical(root/rel)
   if not q.is_relative_to(root.resolve()) or p.sha(q)!=h:raise ValueError('calibration closure')
  if set(c['models'])!=p.cohort_models(c['seeds']):raise ValueError('cohort seed models')
  for cid,m in c['models'].items():
   t=m['training']
   if array_identity is None:array_identity=t['cache_arrays_sha256']
   if t['cache_arrays_sha256']!=array_identity or t['steps']!=32768 or t['positions_seen']!=8388608 or t['unique_cached_positions']!=1048576:raise ValueError('shared teacher/exposure')
   pair=(t['initializer_sha256'],t['initializer_config_sha256'],t['order_sha256'])
   if m['seed'] in inits and inits[m['seed']]!=pair:raise ValueError('paired initialization/order')
   inits[m['seed']]=pair
   for readout in p.READOUTS:
    d=root/'models'/cid/readout;rows=outputs.validate_bundle(d,c,cid,readout,'calibration',rr['dev'],p.GRID)
    zero={(r['dataset'],r['stable_id']):r['prediction'] for r in rows if r['lambda_']==0.}
    if base_zero is None:base_zero=zero
    if zero!=base_zero:raise ValueError('cross-cohort/cal-readout Base zero parity')
    choice=selection.choose(rows)
    policies[p.key(m['seed'],m['objective'],readout)]=dict(seed=m['seed'],objective=m['objective'],readout=readout,checkpoint_sha256=m['files']['model.safetensors']['sha256'],calibration_sha256=p.sha(d/'predictions.jsonl'),completion_sha256=p.sha(d/'COMPLETE.json'),**choice)
  hosts[cohort]=dict(seeds=c['seeds'],config_sha256=c['_sha256'],calibration_complete_sha256=p.sha(root/'CALIBRATION_COMPLETE.json'),training_complete_sha256=c['parent']['training_complete_sha256'])
 selection.validate_policy_keys(policies)
 p.fresh(destination,dict(schema='tec-global-dev-seal-v1',complete=True,source_manifest_sha256=p.source(),population_sha256=pop_sha,cache_arrays_sha256=array_identity,training_source_manifest_sha256=train_source,cohorts=hosts,policies=policies,scientific_acceptance=False))
 open_seal(destination,p.sha(destination))
def main():
 a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);a.add_argument('--cal-root',action='append',required=True)
 for n in ('population-root','output'):a.add_argument('--'+n,required=True)
 a.add_argument('--population-sha256')
 v=a.parse_args();seal(v.cal_root,v.population_root,v.population_sha256 or p.sha(Path(v.population_root)/'COMPLETE.json'),v.output)
if __name__=='__main__':main()
