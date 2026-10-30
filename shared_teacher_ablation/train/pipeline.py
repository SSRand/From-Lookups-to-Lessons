"""Foreground GPU pool: paired training of both objectives for every configured seed.

  python -m shared_teacher_ablation.train.pipeline --config PREP/config/CONFIG.json [--gpus 0,1,...]
"""
import argparse,os,sys,json,time,subprocess,signal,fcntl
from pathlib import Path
from .protocol import *
from . import assets
REPO=HERE.parents[1]

def validate_grant(gpus):
 if not gpus or len(set(gpus))!=len(gpus):raise ValueError('Distinct GPU devices required')

def run_pool(jobs,gpus,logdir,heartbeat=None):
 validate_grant(gpus);active={};pending=list(jobs);done=[];streams=[];beat=0
 if len({j['tag'] for j in jobs})!=len(jobs):raise ValueError('Duplicate worker tag')
 try:
  while pending or active:
   for gpu in gpus:
    if gpu in active or not pending:continue
    job=pending.pop(0);env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    stream=(Path(logdir)/(job['tag']+'.log')).open('x');streams.append(stream)
    p=subprocess.Popen(job['argv'],env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=False,cwd=str(REPO))
    active[gpu]=(p,dict(tag=job['tag'],argv=job['argv'],gpu=gpu,pid=p.pid,started_epoch=time.time()))
   for gpu,(p,r) in list(active.items()):
    if p.poll() is None:continue
    rc=p.wait();r.update(returncode=rc,finished_epoch=time.time());done.append(r);del active[gpu]
    if rc!=0:raise RuntimeError(f'Worker {r["tag"]} failed RC={rc}')
   if heartbeat and time.monotonic()>=beat:heartbeat(done,{g:r for g,(_,r) in active.items()});beat=time.monotonic()+30
   if pending or active:time.sleep(.1)
  return done
 finally:
  for p,_ in active.values():
   if p.poll() is None:p.terminate()
  end=time.monotonic()+10
  for p,_ in active.values():
   try:p.wait(timeout=max(.01,end-time.monotonic()))
   except subprocess.TimeoutExpired:p.kill();p.wait()
  for stream in streams:stream.close()

def main(config_path,config_sha,gpus):
 config_path=str(Path(config_path).absolute());config_sha=config_sha or sha(config_path)
 if sha(config_path)!=config_sha:raise ValueError('Config SHA')
 source_sha=verify_sources()
 config=assets.open_config(config_path,config_sha);config['_config_sha256']=config_sha;out=Path(config['output']).absolute()
 if out.resolve()!=out or out.exists():raise ValueError('Fresh physical training output required')
 validate_grant(gpus);out.mkdir(parents=True);cohort=cells(config['assigned_seeds'])
 lock=(out/'DISPATCH.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def stop(signum,frame):raise KeyboardInterrupt('Foreground training interrupted')
 signal.signal(signal.SIGTERM,stop)
 def status(phase,**kw):
  tmp=out/'.STATUS.tmp';tmp.write_text(json.dumps(dict(phase=phase,epoch=time.time(),source_manifest_sha256=source_sha,config_sha256=config_sha,seeds=config['assigned_seeds'],**kw),indent=2,allow_nan=False)+'\n');os.replace(tmp,out/'STATUS.json')
 def progress(done,active):
  steps={}
  for c in cohort:
   p=out/'trained'/c['id']/'training.jsonl';steps[c['id']]=0
   if p.exists():
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8192));lines=f.read().split(b'\n')
    try:steps[c['id']]=json.loads(lines[-2])['step']
    except (ValueError,KeyError,IndexError):steps[c['id']]=None
  status('training',completed_training=len(done),total_training=len(cohort),worker_steps=steps,active=active)
 dump(out/'START.json',dict(status='started',epoch=time.time(),source_manifest_sha256=source_sha,config_sha256=config_sha,pid=os.getpid(),gpus=gpus,cells=cohort))
 try:
  jobs=[dict(tag=c['id'],argv=[sys.executable,'-B','-m','shared_teacher_ablation.train.worker','--config',config_path,'--expected-config-sha256',config_sha,'--cell',c['id']]) for c in cohort]
  commands=run_pool(jobs,gpus,out,progress);dump(out/'WORKERS_COMPLETE.json',dict(status='complete',commands=commands))
  from .worker import validate_training
  records={c['id']:validate_training(config,c['id'],config_sha) for c in cohort};artifacts={}
  for cid,r in records.items():
   for name in ('COMPLETE.json','AUTH.json',*[x for x in r['artifacts'] if x.endswith(('.json','.jsonl'))]):
    rel=f'trained/{cid}/{name}';artifacts[rel]=sha(out/rel)
  for n in ('START.json','WORKERS_COMPLETE.json'):artifacts[n]=sha(out/n)
  dump(out/'TRAINING_COMPLETE.json',dict(schema='teacher-exposure-host-training-completion-v1',status='complete',complete=True,seeds=config['assigned_seeds'],source_manifest_sha256=source_sha,config_sha256=config_sha,protocol_sha256=canonical(scientific_protocol()),completed_training=len(cohort),total_training=len(cohort),steps_per_model=STEPS,positions_per_model=STEPS*BATCH,cache_arrays_sha256=next(iter(records.values()))['cache_arrays_sha256'],models={cid:dict(completion_sha256=sha(out/'trained'/cid/'COMPLETE.json'),cell=r['cell'],parameter_updates=r['parameter_updates']) for cid,r in records.items()},artifacts=artifacts,quality_evaluation_pending=True,sufficient_convergence=None))
  status('training_complete',complete=True,completed_training=len(cohort),total_training=len(cohort),quality_evaluation_pending=True)
 except BaseException as e:
  dump(out/'FAILURE.json',dict(status='failed_preserved',error=repr(e)));status('failed',complete=False,error=repr(e));raise
 finally:lock.close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--config',required=True);p.add_argument('--expected-config-sha256');p.add_argument('--gpus',help='comma-separated devices; default CUDA_VISIBLE_DEVICES')
 a=p.parse_args();main(a.config,a.expected_config_sha256,a.gpus.split(',') if a.gpus else [x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x])
