"""Matched Base on the 4,096-question diagnostic batches, after the cohort's test phase completes.

  python -m shared_teacher_ablation.corrections.pipeline --config EVAL/CONFIG.json --barrier SEAL.json --output COMP [--gpus 0,1,...]
"""
import argparse,fcntl,os,signal,sys,time
from pathlib import Path
from . import core as c
from ..eval.lifecycle import run_all
p=c.p
REPO=c.ROOT.parents[1]
def command(cfg,a,rank,source,auth_sha,barrier_sha,output):
 return [sys.executable,'-B','-m','shared_teacher_ablation.corrections.worker','--config',cfg['_path'],'--expected-config-sha256',cfg['_sha256'],'--barrier',str(Path(a.barrier).absolute()),'--expected-barrier-sha256',barrier_sha,'--expected-source-manifest-sha256',source,'--expected-test-auth-sha256',auth_sha,'--output',str(output),'--rank',str(rank)]
def main(a):
 source=c.source();gpus=a.gpus.split(',') if a.gpus else [x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
 if not gpus or len(set(gpus))!=len(gpus):raise ValueError('distinct GPU devices required')
 cfg=p.config(str(Path(a.config).absolute()));bs=p.sha(a.barrier)
 c.barrier.open_seal(a.barrier,bs,cfg)
 root=p.physical(a.output);root.mkdir(parents=True,exist_ok=False);lock=(root/'DISPATCH.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def stop(sig,frame):raise InterruptedError('companion signal '+str(sig))
 previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)}
 for s in previous:signal.signal(s,stop)
 def status(phase,**kw):p.atomic(root/'STATUS.json',dict(phase=phase,epoch=time.time(),cohort=cfg['cohort'],source_manifest_sha256=source,**kw))
 try:
  status('authenticating_original_test',complete=False)
  original,records,w=c.original_complete(cfg,bs)
  original_path=Path(cfg['output'])/'test/COMPLETION.json';orig_sha=p.sha(original_path)
  records['COMPLETION.json']=p.record(original_path,orig_sha)
  rr,_=c.population.load_bundle(Path(cfg['population']['COMPLETE.json']['path']).parent,cfg['population']['COMPLETE.json']['sha256']);rr=rr['diagnostic'];plan=c.batch_plan(rr,a.shards)
  if len(rr)!=4096 or len(plan)!=260:raise ValueError('entire fixed diagnostic population')
  for src,dest in ((a.config,'CONFIG.json'),(a.barrier,'GLOBAL_BARRIER.json')):
   with (root/dest).open('xb') as f:f.write(Path(src).read_bytes())
  p.fresh(root/'BATCH_PLAN.json',plan)
  auth=dict(schema='tec-base-diagnostic-auth-v1',source_manifest_sha256=source,cohort=cfg['cohort'],config_sha256=cfg['_sha256'],barrier_sha256=bs,shards=a.shards,original_complete_sha256=orig_sha,batch_plan_sha256=p.sha(root/'BATCH_PLAN.json'),original_records=records)
  p.fresh(root/'TEST_AUTH.json',auth);ah=p.sha(root/'TEST_AUTH.json');tasks=[]
  for rank in range(a.shards):tasks.append(dict(argv=command(cfg,a,rank,source,ah,bs,root),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false'),cwd=str(REPO),log=str(root/(str(rank)+'.log'))))
  def heartbeat(codes):status('base_diagnostic',complete=False,completed_workers=sum(type(x)is int and x==0 for x in codes),total_workers=a.shards)
  rc=run_all(tasks,gpus,heartbeat,30.);p.exact(rc,[0]*a.shards)
  p.fresh(root/'WORKERS_COMPLETE.json',dict(returncodes=rc,commands=[t['argv'] for t in tasks],gpus=gpus))
  for rank in range(a.shards):c.shard(root,cfg,rr,plan,rank,bs,orig_sha,source)
  files={name:p.sha(root/name) for name in sorted(c.companion_artifacts(a.shards))}
  p.fresh(root/'COMPLETION.json',dict(schema='tec-base-diagnostic-completion-v1',status='complete',complete=True,cohort=cfg['cohort'],source_manifest_sha256=source,original_source_manifest_sha256=p.source(),config_sha256=cfg['_sha256'],barrier_sha256=bs,original_complete_sha256=orig_sha,rows=4096,batches=260,workers=a.shards,artifacts=files,scientific_acceptance=False))
  status('base_diagnostic_complete',complete=True,completed_workers=a.shards,total_workers=a.shards,rows=4096)
 except BaseException as e:p.fresh(root/'FAILURE.json',dict(error=repr(e),complete=False,epoch=time.time()));status('failed',complete=False,error=repr(e));raise
 finally:
  for s,v in previous.items():signal.signal(s,v)
  lock.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 for n in ('config','barrier','output'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--shards',type=int,default=8);ap.add_argument('--gpus',help='comma-separated devices; default CUDA_VISIBLE_DEVICES')
 main(ap.parse_args())
