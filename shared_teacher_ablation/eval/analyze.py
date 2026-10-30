"""Offline full-population primary and prespecified crossed-readout diagnostic. No CUDA.

  python -m shared_teacher_ablation.eval.analyze --snapshot EVAL/test [--snapshot ...] --population-root POP --barrier SEAL.json --output ANALYSIS
"""
import argparse,math,time
from pathlib import Path
import numpy as np
from . import protocol as p
from . import population,outputs,barrier
from .paired import aggregate,bootstrap

def array(rows):return np.array([[r['em'],r['f1'],math.fsum(r['token_nll']),len(r['token_nll'])] for r in rows],dtype=float)
def summary(rows):return aggregate([dict(dataset=r['dataset'],exact_match=r['em'],token_f1=r['f1'],nll_sum=math.fsum(r['token_nll']),gold_count=len(r['token_nll'])) for r in rows])
def audit_host(root,rr,b,barrier_sha):
 root=Path(root);done=p.read(root/'COMPLETION.json');c=p.read(root/'CONFIG.json');c['_sha256']=p.sha(root/'CONFIG.json')
 if done['status']!='complete' or done['complete'] is not True or done['phase']!='test' or done['source_manifest_sha256']!=p.source() or done['config_sha256']!=c['_sha256'] or done['barrier_sha256']!=barrier_sha:raise ValueError('completed test capture required')
 if b['cohorts'][c['cohort']]['config_sha256']!=c['_sha256']:raise ValueError('barrier/config')
 for rel,h in done['artifacts'].items():
  q=p.physical(root/rel)
  if not q.is_relative_to(root.resolve()) or p.sha(q)!=h:raise ValueError('exact closure')
 if set(c['models'])!=p.cohort_models(c['seeds']):raise ValueError('all cohort models')
 cells={};bases=[]
 for rank,cid in enumerate(sorted(c['models'])):
  m=c['models'][cid];native=p.NATIVE[m['objective']];other=next(r for r in p.READOUTS if r!=native);w={r:b['policies'][p.key(m['seed'],m['objective'],r)]['selected'] for r in p.READOUTS}
  def get(readout,role,rs,weights,path):return outputs.validate_bundle(path,c,cid,readout,role,rs,weights,barrier_sha)
  primary=get(native,'native_selected',rr['test'],[w[native]],root/'models'/cid/'native_selected')
  br=[r for i,batch in enumerate(outputs.batches(rr['test'])) if i%len(c['models'])==rank for r in batch];bases+=get(native,'base_shard',br,[0.],root/'base'/str(rank))
  crossed=get(other,'cross_selected',rr['diagnostic'],[w[other]],root/'models'/cid/'cross_selected')
  fixed={r:get(r,'fixed1',rr['diagnostic'],[1.],root/'models'/cid/('fixed1-'+r)) for r in p.READOUTS}
  diagnative=get(native,'native_selected_diagnostic',rr['diagnostic'],[w[native]],root/'models'/cid/'native_selected_diagnostic')
  cells[cid]=dict(seed=m['seed'],objective=m['objective'],native=primary,selected={native:diagnative,other:crossed},fixed1=fixed)
 index={r['stable_id']:r for r in bases}
 if set(index)!={r['stable_id'] for r in rr['test']} or len(bases)!=len(rr['test']):raise ValueError('Base full coverage')
 return cells,[index[r['stable_id']] for r in rr['test']],done

def main(a):
 a.population_sha256=a.population_sha256 or p.sha(Path(a.population_root)/'COMPLETE.json');a.expected_barrier_sha256=a.expected_barrier_sha256 or p.sha(a.barrier)
 rr,popdone=population.load_bundle(a.population_root,a.population_sha256);b=barrier.open_seal(a.barrier,a.expected_barrier_sha256);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 cells={};base=None;hostnodes=set();receipts=[]
 for root in a.snapshot:
  cc,bb,done=audit_host(root,rr,b,a.expected_barrier_sha256)
  if done['cohort'] in hostnodes:raise ValueError('duplicate cohort')
  hostnodes.add(done['cohort']);receipts.append(dict(cohort=done['cohort'],completion_sha256=p.sha(Path(root)/'COMPLETION.json')))
  if base is None:base=bb
  else:
   for x,y in zip(base,bb,strict=True):
    for k in ('dataset','stable_id','input_sha256','prediction','token_nll','em','f1','gold_mask_sha256'):p.exact(x[k],y[k])
  if set(cells)&set(cc):raise ValueError('duplicate fitted model')
  cells.update(cc)
 if hostnodes!=set(b['cohorts']) or set(cells)!=p.cohort_models(p.SEEDS):raise ValueError('full16 complete')
 per_seed={};primary={'Base':array(base)};quality={};baseline=primary['Base']
 for o in p.OBJECTIVES:
  series=[]
  for s in p.SEEDS:
   cid=p.cid(s,o);rows=cells[cid]['native'];v=array(rows);series.append(v)
   repair=int(np.sum((baseline[:,0]==0)&(v[:,0]==1)));harm=int(np.sum((baseline[:,0]==1)&(v[:,0]==0)))
   per_seed[cid]=dict(summary=summary(rows),repair_count=repair,harm_count=harm,base_wrong=int(np.sum(baseline[:,0]==0)),base_correct=int(np.sum(baseline[:,0]==1)),repair_rate=repair/int(np.sum(baseline[:,0]==0)),harm_rate=harm/int(np.sum(baseline[:,0]==1)))
  primary[o]=np.mean(series,axis=0)
  quality[o]={k:float(np.mean([per_seed[p.cid(s,o)]['summary']['macro'][k] for s in p.SEEDS])) for k in ('exact_match','token_f1','nll','ppl')}
 def progress(n):p.atomic(out/'STATUS.json',dict(phase='bootstrap',draws=n,epoch=time.time(),complete=False))
 contrasts=[(p.OBJECTIVES[1],p.OBJECTIVES[0]),(p.OBJECTIVES[1],'Base'),(p.OBJECTIVES[0],'Base')]
 ci=bootstrap(rr['test'],primary,contrasts,heartbeat=progress)
 points={a+'-minus-'+bb:{k:quality[a][k]-(summary(base)['macro'][k] if bb=='Base' else quality[bb][k]) for k in ('exact_match','token_f1','nll')} for a,bb in contrasts}
 byseed=[per_seed[p.cid(s,p.OBJECTIVES[1])]['summary']['macro']['token_f1']-per_seed[p.cid(s,p.OBJECTIVES[0])]['summary']['macro']['token_f1'] for s in p.SEEDS]
 secondary={}
 for variant in ('selected','fixed1'):
  arrays={};means={}
  for o in p.OBJECTIVES:
   for r in p.READOUTS:
    label=o+'/'+r;arrays[label]=np.mean([array(cells[p.cid(s,o)][variant][r]) for s in p.SEEDS],axis=0)
    means[label]={k:float(np.mean([summary(cells[p.cid(s,o)][variant][r])['macro'][k] for s in p.SEEDS])) for k in ('exact_match','token_f1','nll','ppl')}
  a00=arrays[p.OBJECTIVES[0]+'/'+p.READOUTS[0]];a01=arrays[p.OBJECTIVES[0]+'/'+p.READOUTS[1]];a10=arrays[p.OBJECTIVES[1]+'/'+p.READOUTS[0]];a11=arrays[p.OBJECTIVES[1]+'/'+p.READOUTS[1]]
  plus=a11.copy();plus[:,:3]+=a00[:,:3];minus=a10.copy();minus[:,:3]+=a01[:,:3];arrays.update(interaction_plus=plus,interaction_minus=minus)
  dc=bootstrap(rr['diagnostic'],arrays,[('interaction_plus','interaction_minus')],heartbeat=progress)
  secondary[variant]=dict(means=means,interaction_ci=dc,scope='balanced4096 secondary; calibrated system interaction for selected policies')
 result=dict(status='computed',scientific_acceptance=False,primary_population=38627,primary_metric='equal_dataset_macro_F1_RES_minus_MLP',full_seed_count=8,primary=quality,base=summary(base),per_seed=per_seed,paired_seed_F1_differences=byseed,paired_seed_sample_sd=float(np.std(byseed,ddof=1)),primary_CIs=ci,primary_contrast_points=points,secondary=secondary,policies=b['policies'],source_manifest_sha256=p.source(),barrier_sha256=a.expected_barrier_sha256,population_sha256=a.population_sha256,cohort_completions=receipts,limits=['question-cluster intervals conditional on fitted seeds/policies; seed variation separately','same continuation teacher/exposure but objective and native readout jointly differ','no equivalence or universal convergence claim','gold NLL aggregated from stored full masks, no independent model re-inference'])
 p.fresh(out/'ANALYSIS.json',result);p.fresh(out/'OFFLINE_RESULT.json',dict(status='computed',scientific_acceptance=False,analysis_sha256=p.sha(out/'ANALYSIS.json'),source_manifest_sha256=p.source(),barrier_sha256=a.expected_barrier_sha256,cohort_completions=receipts))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--snapshot',action='append',required=True)
 for n in ('population-root','barrier','output'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--population-sha256');ap.add_argument('--expected-barrier-sha256')
 main(ap.parse_args())
