"""One foreground memory worker; shared teacher, exact paired order and dose."""
import argparse,json,math,os,random,time,shutil
from pathlib import Path
import numpy as np
from .protocol import *
from . import assets

def identify(config,cid,config_sha):
 c=next((x for x in cells(config['assigned_seeds']) if x['id']==cid),None)
 if c is None:raise ValueError('Cell outside configured cohort')
 _,order_sha=orders(N,EPOCHS,c['seed']);init=config['initializers'][str(c['seed'])]
 return dict(cell=c,source_manifest_sha256=verify_sources(),config_sha256=config_sha,protocol_sha256=canonical(scientific_protocol()),cache_complete_sha256=config['cache_complete_sha256'],cache_arrays_sha256=canonical({k:v['sha256'] for k,v in config['cache_producer']['arrays'].items()}),initializer_sha256=init['model_sha256'],initializer_config_sha256=init['config_sha256'],order_sha256=order_sha,steps=STEPS,positions_seen=STEPS*BATCH,unique_cached_positions=N,parameter_count=parameter_count(8),probe_indices_sha256=config['probe_indices_sha256'])

def validate_trace(records,total=STEPS):
 if len(records)!=total:raise ValueError('Incomplete update trace')
 for i,r in enumerate(records):
  if type(r.get('step')) is not int or r['step']!=i+1 or type(r.get('positions_seen')) is not int or r['positions_seen']!=(i+1)*BATCH or r.get('lr')!=learning_rate(i,total):raise ValueError('Dose/schedule trace mismatch')
  for k in ('loss','kl','ce','grad_norm','elapsed_seconds'):
   if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]):raise ValueError('Nonfinite training trace')
  wanted=(i+1)%64==0
  if wanted!=(r.get('update_sample') is not None):raise ValueError('Parameter update sample coverage')
  if wanted:
   u=r['update_sample']
   if u.get('finite') is not True or not math.isfinite(u['l2']) or u['l2']<0 or not all(math.isfinite(x) and x>=0 for x in u['parameters'].values()):raise ValueError('Invalid actual update sample')
 return dict(samples=total//64,finite=True,nonzero_samples=sum(r['update_sample']['l2']>0 for r in records if r['update_sample'] is not None),scope='each64th update; fixed64 entries per tensor; diagnostic only')

def expected_artifacts():
 steps=[e*(N//BATCH) for e in MILESTONE_EPOCHS]
 return {'training.jsonl','optimizer_rng.pt','RECOVERY_STATE.json','FIRST64_ETA.json',*[f'probes/{s}.json' for s in [0,*steps]],*[f'milestones/{s}/{n}' for s in steps for n in ('COMPLETE.json','checkpoint/model.safetensors','checkpoint/config.json')]}

def validate_training(config,cid,config_sha=None):
 if config_sha is None:config_sha=config['_config_sha256']
 d=Path(config['output'])/'trained'/cid;r=read(d/'COMPLETE.json');wanted=identify(config,cid,config_sha);auth=read(d/'AUTH.json')
 if r.get('schema')!='teacher-exposure-model-completion-v1' or r.get('status')!='complete' or any(type(r.get(k)) is not type(v) or r[k]!=v for k,v in wanted.items()):raise ValueError('Model completion identity')
 if auth['completion_sha256']!=sha(d/'COMPLETE.json') or auth['source_manifest_sha256']!=wanted['source_manifest_sha256']:raise ValueError('Writer authentication changed')
 if set(r['artifacts'])!=expected_artifacts():raise ValueError('Exact checkpoint/probe/state artifact closure required')
 paths=[d/'COMPLETE.json']
 for rel,h in r['artifacts'].items():
  p=assets.physical_child(d,rel);paths.append(p)
  if p.stat().st_size<=32*1024**2 and sha(p)!=h:raise ValueError('Small completion payload mismatch')
 assets.verify_snapshot(paths,auth['stats'])
 rec=rows(d/'training.jsonl');updates=validate_trace(rec)
 if r['parameter_updates']!=updates:raise ValueError('Update summary mismatch')
 for epoch in MILESTONE_EPOCHS:
  step=epoch*(N//BATCH);cp=read(d/f'milestones/{step}/COMPLETE.json')
  if cp['step']!=step or cp['epoch']!=epoch or cp['training_identity']!=wanted:raise ValueError('Milestone lineage mismatch')
  if set(cp['checkpoint_artifacts'])!={f'milestones/{step}/checkpoint/model.safetensors',f'milestones/{step}/checkpoint/config.json'}:raise ValueError('Milestone payload coverage')
  for rel,h in cp['checkpoint_artifacts'].items():
   if r['artifacts'].get(rel)!=h:raise ValueError('Milestone model edge differs')
 return r

def train(config_path,config_sha,cid,resume_from=None,resume_sha=None,destination=None):
 if sha(config_path)!=config_sha:raise ValueError('Config hash')
 config=assets.open_config(config_path,config_sha);config['_config_sha256']=config_sha
 import torch
 from resmem.memory import MistralMLPModel
 from . import numeric
 wanted=identify(config,cid,config_sha);c=wanted['cell'];out=Path(config['output'])/'trained'/cid
 if any(x is not None for x in (resume_from,resume_sha,destination)):
  if not all(x is not None for x in (resume_from,resume_sha,destination)):raise ValueError('Explicit resume path/SHA/fresh destination required')
  out=Path(destination).absolute()
  if out.parent!=Path(config['output'])/'resume-attempts'/cid:raise ValueError('Recovery destination outside private cell')
 if out.exists() or out.resolve()!=out:raise ValueError('Fresh physical worker output required')
 init=config['initializers'][str(c['seed'])];model_path=Path(init['path']);state=None;prior=[];resume_root=None
 if resume_from:
  p=Path(resume_from).absolute();resume_root=p.parent
  if p.resolve()!=p or p.name!='optimizer_rng.pt' or sha(p)!=resume_sha:raise ValueError('Resume payload identity')
  pointer=read(resume_root/'RECOVERY_STATE.json')
  if pointer['optimizer_rng_sha256']!=resume_sha or pointer['training_identity']!=wanted:raise ValueError('Resume pointer/identity')
  for rel,h in pointer['checkpoint_artifacts'].items():
   if sha(assets.physical_child(resume_root,rel))!=h:raise ValueError('Resume model hash')
  state=torch.load(p,map_location='cpu',weights_only=False)
  if state['checkpoint_artifacts']!=pointer['checkpoint_artifacts']:raise ValueError('Resume model/state edge')
  model_path=resume_root/next(x for x in pointer['checkpoint_artifacts'] if x.endswith('model.safetensors'));model_path=model_path.parent
  prior=rows(resume_root/'training.jsonl')[:state['step']]
  if len(prior)!=state['step'] or canonical(prior)!=state['training_prefix_sha256']:raise ValueError('Resume trace prefix')
 torch.set_num_threads(4);random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);torch.backends.cuda.matmul.allow_tf32=False
 model,info=MistralMLPModel.from_pretrained(model_path,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('Strict memory checkpoint load')
 if len(model.layers)!=8 or sum(x.numel() for x in model.parameters())!=parameter_count(8) or {x.dtype for x in model.parameters()}!={torch.float32}:raise ValueError('Native architecture/precision')
 model=model.cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.01);arrays=assets.open_arrays(config);all_orders,order_sha=orders(N,EPOCHS,c['seed'])
 step=0;artifacts={};out.mkdir(parents=True);started=time.monotonic();prior_seconds=prior[-1]['elapsed_seconds'] if prior else 0
 if state is not None:
  step=numeric.restore_state(opt,state,wanted,order_sha,N,BATCH,STEPS)
  for rel,h in state['prior_artifacts'].items():
   src=assets.physical_child(resume_root,rel)
   if sha(src)!=h:raise ValueError('Resume prior artifact changed')
   target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
   # Explicit recovery copies history; even inode ctime of the failed source stays unchanged.
   shutil.copyfile(src,target);artifacts[rel]=h
  dump(out/'RESUME_PROVENANCE.json',dict(status='explicit_resume',state_path=resume_from,state_sha256=resume_sha,step=step))
 def probe(s):
  sums=np.zeros(3);indices=config['probe_indices'];t=time.monotonic();model.eval()
  with torch.no_grad():
   for j in range(0,len(indices),MICRO):
    vv=numeric.forward(model,numeric.batch(arrays,indices[j:j+MICRO],'cuda'),c['objective'],ALPHA);z=np.array([float(v) for v in vv])
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite train probe')
    sums+=z*MICRO
  model.train();p=out/f'probes/{s}.json';dump(p,dict(step=s,epoch=s/(N//BATCH),positions=len(indices),total=float(sums[0]/len(indices)),kl=float(sums[1]/len(indices)),ce=float(sums[2]/len(indices)),probe_indices_sha256=config['probe_indices_sha256'],scope='fixed training-position diagnostic; not QA or independent validation',elapsed_seconds=time.monotonic()-t));artifacts[str(p.relative_to(out))]=sha(p)
 def save(s):
  if any(not bool(torch.isfinite(v).all()) for v in model.parameters()):raise FloatingPointError('Nonfinite checkpoint parameter')
  checkpoint=out/f'milestones/{s}/checkpoint';model.save_pretrained(checkpoint,safe_serialization=True,max_shard_size='10GB')
  hashes={str((checkpoint/n).relative_to(out)):sha(checkpoint/n) for n in ('model.safetensors','config.json')};artifacts.update(hashes)
  record=out/f'milestones/{s}/COMPLETE.json';dump(record,dict(status='complete',step=s,epoch=s//(N//BATCH),training_identity=wanted,checkpoint_artifacts=hashes,probe_sha256=sha(out/f'probes/{s}.json')));artifacts[str(record.relative_to(out))]=sha(record)
  st=numeric.save_state(opt,s,wanted,order_sha,N,BATCH,copy_optimizer=False);st['checkpoint_artifacts']=hashes;st['training_prefix_sha256']=canonical(rows(out/'training.jsonl'));st['prior_artifacts']=dict(artifacts)
  tmp=out/'optimizer_rng.pt.partial';torch.save(st,tmp);os.replace(tmp,out/'optimizer_rng.pt');del st
  pointer=dict(status='current_recovery_state',step=s,optimizer_rng_sha256=sha(out/'optimizer_rng.pt'),checkpoint_artifacts=hashes,training_identity=wanted)
  tmp=out/'RECOVERY_STATE.json.partial';tmp.write_text(json.dumps(pointer,indent=2,allow_nan=False)+'\n');os.replace(tmp,out/'RECOVERY_STATE.json')
 try:
  if state is None:probe(0)
  resume_step=step
  with (out/'training.jsonl').open('x') as log:
   for r in prior:log.write(json.dumps(r,allow_nan=False)+'\n')
   for epoch,order in enumerate(all_orders):
    for start in range(0,N,BATCH):
     if epoch*(N//BATCH)+start//BATCH<resume_step:continue
     v=numeric.step_update(model,opt,numeric.batch(arrays,order[start:start+BATCH],'cuda'),c['objective'],step,STEPS,MICRO,ALPHA,sample=(step+1)%64==0);step+=1
     rec=dict(step=step,positions_seen=step*BATCH,elapsed_seconds=prior_seconds+time.monotonic()-started,**v);log.write(json.dumps(rec,allow_nan=False)+'\n');log.flush()
     if step%32==0:print(json.dumps(dict(cell=cid,**rec),allow_nan=False),flush=True)
     if step==64:
      dump(out/'FIRST64_ETA.json',dict(step=64,elapsed_seconds=rec['elapsed_seconds'],projected_train_seconds=rec['elapsed_seconds']/64*STEPS,scope='observed first64 updates inclinitial probe/load excluded; projection only, no budget mutation'))
      artifacts['FIRST64_ETA.json']=sha(out/'FIRST64_ETA.json')
     if step in [e*(N//BATCH) for e in MILESTONE_EPOCHS]:probe(step);save(step)
  records=rows(out/'training.jsonl');updates=validate_trace(records)
  for name in ('training.jsonl','optimizer_rng.pt','RECOVERY_STATE.json','FIRST64_ETA.json'):artifacts[name]=sha(out/name)
  dump(out/'COMPLETE.json',dict(schema='teacher-exposure-model-completion-v1',status='complete',**wanted,parameter_updates=updates,elapsed_seconds=prior_seconds+time.monotonic()-started,artifacts=artifacts,stationarity='diagnostic probes only; not certified convergence'))
  paths=[out/'COMPLETE.json',*[out/r for r in artifacts]];dump(out/'AUTH.json',dict(status='complete',source_manifest_sha256=wanted['source_manifest_sha256'],completion_sha256=sha(out/'COMPLETE.json'),stats=assets.snapshot(paths)))
  if resume_from is None:validate_training(config,cid,config_sha)
 except BaseException as e:
  dump(out/'FAILURE.json',dict(status='failed_preserved',step=step,error=repr(e)));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--expected-config-sha256');p.add_argument('--cell',required=True);p.add_argument('--resume-from');p.add_argument('--resume-sha');p.add_argument('--destination');a=p.parse_args();train(a.config,a.expected_config_sha256 or sha(a.config),a.cell,a.resume_from,a.resume_sha,a.destination)
