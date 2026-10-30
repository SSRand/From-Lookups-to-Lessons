"""Separate calibration (dev grid) and test phases; one foreground worker per model on a GPU pool.

  python -m shared_teacher_ablation.eval.pipeline --config EVAL/CONFIG.json --phase calibration [--gpus 0,1,...]
  python -m shared_teacher_ablation.eval.pipeline --config EVAL/CONFIG.json --phase test --barrier SEAL.json [--gpus ...]
"""
import argparse,fcntl,os,signal,sys,time
from pathlib import Path
from . import protocol as p
from . import population,outputs,barrier
from .lifecycle import run_all
REPO=p.ROOT.parents[1]

def main(a):
 gpus=a.gpus.split(',') if a.gpus else [x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
 if not gpus or len(set(gpus))!=len(gpus):raise ValueError('distinct GPU devices required')
 config=str(Path(a.config).absolute());c=p.config(config,a.expected_config_sha256);config_sha=c['_sha256'];root=Path(c['output']);phase=root/a.phase;phase.mkdir(parents=True,exist_ok=False)
 barrier_sha=p.sha(a.barrier) if a.phase=='test' else None
 lock=(phase/'DISPATCH.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def stop(sig,frame):raise KeyboardInterrupt('foreground evaluation stopped')
 signal.signal(signal.SIGTERM,stop)
 def status(state,**kw):p.atomic(root/'STATUS.json',dict(phase=state,epoch=time.time(),cohort=c['cohort'],source_manifest_sha256=c['source_manifest_sha256'],config_sha256=config_sha,**kw))
 n=len(c['models'])
 try:
  if a.phase=='test':barrier.open_seal(a.barrier,barrier_sha,c)
  elif a.barrier:raise ValueError('no test seal during calibration')
  with (phase/'CONFIG.json').open('xb') as f:f.write(Path(config).read_bytes())
  if a.phase=='test':
   with (phase/'GLOBAL_BARRIER.json').open('xb') as f:f.write(Path(a.barrier).read_bytes())
  tasks=[]
  for rank,cid in enumerate(sorted(c['models'])):
   argv=[sys.executable,'-B','-m','shared_teacher_ablation.eval.worker','--config',config,'--expected-config-sha256',config_sha,'--cell',cid,'--rank',str(rank),'--phase',a.phase]
   if a.phase=='test':argv+=['--barrier',str(Path(a.barrier).absolute()),'--expected-barrier-sha256',barrier_sha]
   tasks.append(dict(argv=argv,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),cwd=str(REPO),log=str(phase/(cid+'.log'))))
  def heartbeat(codes):
   done=sum(x is not None and x==0 for x in codes);counts={}
   for cid in c['models']:
    q=phase/(cid+'.STATUS.json')
    if q.exists():counts[cid]=p.read(q)
   status(a.phase,complete=False,completed_models=done,total_models=n,workers=counts)
  rc=run_all(tasks,gpus,heartbeat,30.)
  p.fresh(phase/'WORKERS_COMPLETE.json',dict(returncodes=rc,commands=[t['argv'] for t in tasks],gpus=gpus))
  rr,_=population.load_bundle(Path(c['population']['COMPLETE.json']['path']).parent,c['population']['COMPLETE.json']['sha256']);files={'CONFIG.json':p.sha(phase/'CONFIG.json'),'WORKERS_COMPLETE.json':p.sha(phase/'WORKERS_COMPLETE.json')}
  b=barrier.open_seal(a.barrier,barrier_sha,c) if a.phase=='test' else None
  total=0
  for rank,cid in enumerate(sorted(c['models'])):
   m=c['models'][cid];native=p.NATIVE[m['objective']];other=next(r for r in p.READOUTS if r!=native)
   if a.phase=='calibration':roles=[(r,'calibration',rr['dev'],p.GRID,phase/'models'/cid/r) for r in p.READOUTS]
   else:
    w={r:b['policies'][p.key(m['seed'],m['objective'],r)]['selected'] for r in p.READOUTS}
    br=[r for i,batch in enumerate(outputs.batches(rr['test'])) if i%n==rank for r in batch]
    roles=[(native,'native_selected',rr['test'],[w[native]],phase/'models'/cid/'native_selected'),(native,'base_shard',br,[0.],phase/'base'/str(rank)),(native,'native_selected_diagnostic',rr['diagnostic'],[w[native]],phase/'models'/cid/'native_selected_diagnostic'),(other,'cross_selected',rr['diagnostic'],[w[other]],phase/'models'/cid/'cross_selected')]+[(r,'fixed1',rr['diagnostic'],[1.],phase/'models'/cid/('fixed1-'+r)) for r in p.READOUTS]
   for r,role,pop,weights,d in roles:
    total+=len(outputs.validate_bundle(d,c,cid,r,role,pop,weights,barrier_sha if b else None))
    for name in ('COMPLETE.json','predictions.jsonl'):files[str((d/name).relative_to(phase))]=p.sha(d/name)
  if b:files['GLOBAL_BARRIER.json']=p.sha(phase/'GLOBAL_BARRIER.json')
  receipt=dict(schema='tec-evaluation-phase-v1',status='complete',complete=True,phase=a.phase,cohort=c['cohort'],seeds=c['seeds'],source_manifest_sha256=c['source_manifest_sha256'],config_sha256=config_sha,training_complete_sha256=c['parent']['training_complete_sha256'],population_sha256=c['population']['COMPLETE.json']['sha256'],barrier_sha256=barrier_sha if b else None,completed_models=n,policies=2*n if not b else 32,rows=total,artifacts=files,scientific_acceptance=False)
  p.fresh(phase/('CALIBRATION_COMPLETE.json' if not b else 'COMPLETION.json'),receipt);status(a.phase+'_complete',complete=True,completed_models=n,total_models=n,rows=total,scientific_acceptance=False)
 except BaseException as e:p.fresh(phase/'FAILURE.json',dict(error=repr(e),epoch=time.time(),complete=False));status('failed',complete=False,error=repr(e));raise
 finally:lock.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter);ap.add_argument('--config',required=True);ap.add_argument('--expected-config-sha256');ap.add_argument('--phase',choices=['calibration','test'],required=True);ap.add_argument('--barrier');ap.add_argument('--gpus',help='comma-separated devices; default CUDA_VISIBLE_DEVICES');main(ap.parse_args())
