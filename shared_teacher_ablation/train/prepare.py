"""One CPU full cache/initializer authentication; produces the training CONFIG."""
import argparse,time
from pathlib import Path
from .protocol import *
from . import assets

def prepare(request_path,request_sha,output):
 import torch
 started=time.monotonic();source=verify_sources()
 if torch.cuda.is_initialized():raise RuntimeError('CPU prepare must not initialize CUDA')
 if sha(request_path)!=request_sha:raise ValueError('Preparation request changed')
 r=read(request_path);seeds=tuple(r['seeds']);cells(seeds);out=Path(output).absolute()
 if out.resolve()!=out or out.exists():raise ValueError('Fresh physical preparation directory required')
 if set(r['initializers'])!={str(s) for s in seeds}:raise ValueError('Exact assigned initializers required')
 paths=[Path(request_path),Path(r['cache_complete_path']),Path(r['training_jsonl_path'])]
 cache,files=assets.cache_files(r['cache_complete_path'],r['cache_complete_sha256'],r['training_jsonl_path'],r['training_jsonl_sha256']);paths+=list(files.values())
 init={}
 for seed in seeds:
  rec=r['initializers'][str(seed)];directory=Path(rec['path']).absolute()
  for name in ('config.json','model.safetensors','COMPLETE.json'):paths.append(directory/name)
  init[str(seed)]=dict(rec,path=str(directory))
 before=assets.snapshot(paths)
 if sha(r['training_jsonl_path'])!=r['training_jsonl_sha256'] or sha(r['cache_complete_path'])!=r['cache_complete_sha256']:raise ValueError('Producer changed before protected validation')
 arrays={}
 for name,path in files.items():
  rec=cache['arrays'][name]
  if sha(path)!=rec['sha256']:raise ValueError('Cache payload SHA: '+name)
  arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
  if list(arrays[name].shape)!=rec['shape'] or str(arrays[name].dtype)!=rec['dtype']:raise ValueError('Manifest array metadata: '+name)
 training=rows(r['training_jsonl_path'])
 if len(training)!=N//32 or any(len(x['target_ids'])!=32 for x in training):raise ValueError('Training population/targets')
 assets.validate_arrays(arrays,training)
 for seed,rec in init.items():
  p=Path(rec['path'])
  for name,key in [('model.safetensors','model_sha256'),('config.json','config_sha256'),('COMPLETE.json','receipt_sha256')]:
   if sha(p/name)!=rec[key]:raise ValueError('Initializer requested identity mismatch')
  hashes=assets.validate_initializer(p,int(seed));rec['tensor_sha256']=hashes
  original=read(p/'COMPLETE.json')
  if original.get('status')!='complete' or original.get('seed',int(seed))!=int(seed) or original.get('tensor_sha256')!=hashes:raise ValueError('Initializer producer receipt mismatch')
 assets.verify_snapshot(paths,before)
 if torch.cuda.is_initialized():raise RuntimeError('CPU prep touched CUDA')
 out.mkdir(parents=True)
 c=dict(schema='teacher-exposure-host-config-v1',status='complete',assigned_seeds=list(seeds),source_manifest_sha256=source,protocol=scientific_protocol(),software=assets.software(),cache_complete_path=r['cache_complete_path'],cache_complete_sha256=r['cache_complete_sha256'],cache_arrays={k:str(v) for k,v in files.items()},cache_producer=cache,training_jsonl_path=r['training_jsonl_path'],training_jsonl_sha256=r['training_jsonl_sha256'],initializers=init,immutable_stats=before,probe_indices=probe_indices(),probe_indices_sha256=canonical(probe_indices()),output=r['output'],request_sha256=request_sha)
 dump(out/'CONFIG.json',c);dump(out/'CPU_COMPLETE.json',dict(status='complete',config_sha256=sha(out/'CONFIG.json'),source_manifest_sha256=source,elapsed_seconds=time.monotonic()-started,cuda_initialized=False,files=len(paths),scope='All array SHA/shape/finite/causal gold and exact fresh initializer tensors'))
 return c

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--expected-request-sha256');p.add_argument('--output',required=True);a=p.parse_args();prepare(a.request,a.expected_request_sha256 or sha(a.request),a.output)
