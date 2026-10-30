"""Exploratory four-cell correction/preservation on the diagnostic set; completed captures only.

  python -m shared_teacher_ablation.corrections.analyze_corrections --cohort EVAL/test COMP [--cohort ...] --population-root POP --barrier SEAL.json --output CORR
"""
import argparse,math,os,time
from pathlib import Path
import numpy as np
from . import core as c
from . import statistics_core as s
p=c.p

def companion(root,cfg,rr,bs,original_sha,src):
 root=p.physical(root);d=p.read(root/'COMPLETION.json')
 want=dict(schema='tec-base-diagnostic-completion-v1',status='complete',complete=True,cohort=cfg['cohort'],source_manifest_sha256=src,original_source_manifest_sha256=p.source(),config_sha256=cfg['_sha256'],barrier_sha256=bs,original_complete_sha256=original_sha,rows=4096,batches=260,scientific_acceptance=False)
 for k,v in want.items():p.exact(d[k],v)
 shards=d['workers']
 c.artifact_records(root,d['artifacts'],c.companion_artifacts(shards));p.exact(p.sha(root/'CONFIG.json'),cfg['_sha256']);p.exact(p.sha(root/'GLOBAL_BARRIER.json'),bs)
 auth=p.read(root/'TEST_AUTH.json');plan=c.batch_plan(rr,shards);p.exact(p.read(root/'BATCH_PLAN.json'),plan)
 for k,w in dict(schema='tec-base-diagnostic-auth-v1',source_manifest_sha256=src,cohort=cfg['cohort'],config_sha256=cfg['_sha256'],barrier_sha256=bs,shards=shards,original_complete_sha256=original_sha,batch_plan_sha256=p.sha(root/'BATCH_PLAN.json')).items():p.exact(auth[k],w)
 w=p.read(root/'WORKERS_COMPLETE.json');p.exact(w['returncodes'],[0]*shards)
 if len(w['commands'])!=shards:raise ValueError('exact worker commands')
 rows=[]
 for rank in range(shards):
  cmd=w['commands'][rank]
  if cmd[3]!='shared_teacher_ablation.corrections.worker' or cmd[-2:]!=['--rank',str(rank)] or cmd[cmd.index('--expected-test-auth-sha256')+1]!=p.sha(root/'TEST_AUTH.json'):raise ValueError('companion worker argv')
  r,receipt=c.shard(root,cfg,rr,plan,rank,bs,original_sha,src);rows.extend(r)
 index={x['stable_id']:x for x in rows}
 if len(index)!=4096 or len(rows)!=4096:raise ValueError('complete unique Base diagnostic')
 return [index[r['stable_id']] for r in rr],d,auth

def load_host(original,root,rr,seal,bs,src):
 original=p.physical(original);root=p.physical(root);origsha=p.sha(original/'COMPLETION.json')
 cfg=p.read(original/'CONFIG.json');cfg['_sha256']=p.sha(original/'CONFIG.json')
 p.exact(cfg['source_manifest_sha256'],p.source());p.exact(cfg['runtime'],p.RUNTIME)
 if set(cfg['models'])!=p.cohort_models(cfg['seeds']):raise ValueError('full cohort models')
 p.exact(seal['cohorts'][cfg['cohort']]['config_sha256'],cfg['_sha256']);p.exact(cfg['population']['COMPLETE.json']['sha256'],seal['population_sha256'])
 orig,records,workers=c.original_complete(cfg,bs,root=original)
 base,done,auth=companion(root,cfg,rr,bs,origsha,src)
 if set(auth['original_records'])!=set(orig['artifacts'])|{'COMPLETION.json'}:raise ValueError('original authentication exact scope')
 for rel,h in dict(orig['artifacts'],**{'COMPLETION.json':origsha}).items():
  p.exact(auth['original_records'][rel]['sha256'],h);p.exact(auth['original_records'][rel]['path'],str(original/rel))
 arms={}
 for cid,m in cfg['models'].items():
  native=p.NATIVE[m['objective']]
  for readout in p.READOUTS:
   policy=seal['policies'][p.key(m['seed'],m['objective'],readout)];p.exact(policy['checkpoint_sha256'],m['files']['model.safetensors']['sha256'])
   role='native_selected_diagnostic' if readout==native else 'cross_selected'
   for variant,r,w,path in [('selected',role,policy['selected'],original/'models'/cid/role),('fixed1','fixed1',1.,original/'models'/cid/('fixed1-'+readout))]:
    key=f'{variant}/{m["objective"]}/{readout}/{m["seed"]}'
    if key in arms:raise ValueError('duplicate arm')
    arms[key]=c.outputs.validate_bundle(path,cfg,cid,readout,r,rr,[w],bs)
 return cfg['cohort'],base,arms,dict(original_complete_sha256=origsha,companion_complete_sha256=p.sha(root/'COMPLETION.json'))

def parity(a,b):
 if len(a)!=len(b):raise ValueError('cross-cohort Base coverage')
 for x,y in zip(a,b,strict=True):
  for k in ('dataset','stable_id','input_sha256','gold_mask_sha256','prediction','token_nll','em','f1','lambda_','global_batch_id','geometry_sha256'):p.exact(x[k],y[k])

def main(a):
 if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('explicit CPU-only analysis: set CUDA_VISIBLE_DEVICES=""')
 src=c.source();pop_sha=p.sha(Path(a.population_root)/'COMPLETE.json');bs=p.sha(a.barrier)
 population,_=c.population.load_bundle(a.population_root,pop_sha);rr=population['diagnostic'];seal=c.barrier.open_seal(a.barrier,bs)
 p.exact(seal['population_sha256'],pop_sha)
 base=None;allarms={};hosts={}
 for original,comp in a.cohort:
  node,bb,arms,proof=load_host(original,comp,rr,seal,bs,src)
  if node in hosts or set(allarms)&set(arms):raise ValueError('duplicate cohort/arm')
  if base is None:base=bb
  else:parity(base,bb)
  hosts[node]=proof;allarms.update(arms)
 expected={f'{v}/{o}/{r}/{seed}' for v in ('selected','fixed1') for o in p.OBJECTIVES for r in p.READOUTS for seed in p.SEEDS}
 if set(hosts)!=set(seal['cohorts']) or set(allarms)!=expected:raise ValueError('full64 diagnostic roles')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 try:
  per_seed={};means={};arrays={};bins={}
  for name,rows in allarms.items():per_seed[name]=s.summaries(rr,base,rows);bins[name]=s.token_bins(rr,base,rows)
  for v in ('selected','fixed1'):
   for o in p.OBJECTIVES:
    for r in p.READOUTS:
     label=f'{v}/{o}/{r}';keys=[label+'/'+str(seed) for seed in p.SEEDS]
     arrays[label]=(np.mean([[x['em'] for x in allarms[k]] for k in keys],axis=0),np.mean([[x['f1'] for x in allarms[k]] for k in keys],axis=0))
     means[label]={}
     for ds in sorted(p.COUNTS)+['macro']:
      views=[per_seed[k]['macro'] if ds=='macro' else per_seed[k]['datasets'][ds] for k in keys]
      means[label][ds]={m:None if any(x[m] is None for x in views) else math.fsum(x[m] for x in views)/len(views) for m in s.METRICS}
  def heartbeat(n):p.atomic(out/'STATUS.json',dict(phase='paired_ratio_bootstrap',draws=n,complete=False,epoch=time.time()))
  ci=s.bootstrap(rr,np.array([x['em'] for x in base]),np.array([x['f1'] for x in base]),arrays,heartbeat=heartbeat)
  result=dict(schema='tec-diagnostic-corrections-v1',status='computed',scientific_acceptance=False,descriptive_exploratory=True,population=4096,seeds=list(p.SEEDS),roles=64,source_manifest_sha256=src,original_source_manifest_sha256=p.source(),population_sha256=pop_sha,barrier_sha256=bs,cohort_completions=hosts,per_seed=per_seed,seed_mean_points=means,question_cluster_CIs=ci,token_probability_bins=bins,policies=seal['policies'],limits=['all selected/fixed1 four cells retained; exploratory multiple views, no multiplicity-adjusted confirmatory claims','conditional question-cluster intervals; fitted seeds/policies fixed','normalized alias EM is not semantic adjudication','gold probability is canonical teacher-forced token probability, not semantic confidence','Base-conditioned bins do not identify knowledge origin or causal mediation','no primary or model/policy retuning'])
  p.fresh(out/'ANALYSIS.json',result);p.fresh(out/'OFFLINE_RESULT.json',dict(status='computed',scientific_acceptance=False,analysis_sha256=p.sha(out/'ANALYSIS.json'),source_manifest_sha256=src,barrier_sha256=bs,cohort_completions=hosts));p.atomic(out/'STATUS.json',dict(phase='computed',complete=True,scientific_acceptance=False,epoch=time.time()))
 except BaseException as e:p.fresh(out/'FAILURE.json',dict(error=repr(e),epoch=time.time()));raise
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);ap.add_argument('--cohort',nargs=2,action='append',required=True,metavar=('TEST_PHASE_ROOT','COMPANION_ROOT'))
 for n in ('population-root','barrier','output'):ap.add_argument('--'+n,required=True)
 main(ap.parse_args())
