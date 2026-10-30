"""Finite CPU-only initializer stage followed by completed-cache preparation.

  python -m shared_teacher_ablation.train.cpu_stage --cache CACHE --data DATA --output PREP --training-output TRAIN [--seeds 42,123,...]
writes PREP/initializers/<seed>, PREP/REQUEST.json and PREP/config/CONFIG.json.
"""
import argparse,time,os,threading
from pathlib import Path
from .protocol import *
from .initialize import create

def initializers(seeds,output,source):
 verify_sources(source);out=Path(output).absolute()
 if out.exists() or out.resolve()!=out:raise ValueError('Fresh physical CPU stage required')
 cells(seeds)
 out.mkdir(parents=True);records={};stop=threading.Event();state=dict(phase='initializers',seeds=list(seeds),completed_initializers=0,total_initializers=len(seeds),source_manifest_sha256=source)
 def status():
  tmp=out/'.STATUS.tmp';tmp.write_text(json.dumps(dict(state,epoch=time.time()),indent=2)+'\n');os.replace(tmp,out/'STATUS.json')
 def beat():
  while not stop.wait(30):status()
 status();thread=threading.Thread(target=beat,daemon=True);thread.start();started=time.monotonic()
 try:
  for seed in seeds:
   state['current_seed']=seed;p=out/'initializers'/str(seed);create(seed,p);r=read(p/'COMPLETE.json')
   if r['status']!='complete' or r['seed']!=seed:raise ValueError('Initializer completion differs')
   records[str(seed)]=dict(path=str(p),model_sha256=r['artifacts']['model.safetensors'],config_sha256=r['artifacts']['config.json'],receipt_sha256=sha(p/'COMPLETE.json'));state['completed_initializers']=len(records)
  result=dict(schema='teacher-exposure-host-initializers-v1',status='complete',seeds=list(seeds),source_manifest_sha256=source,initializers=records,elapsed_seconds=time.monotonic()-started)
  dump(out/'INITIALIZERS_COMPLETE.json',result);state.update(phase='initializers_complete',complete=True);return result
 except BaseException as e:
  state.update(phase='failed',error=repr(e));dump(out/'FAILURE.json',dict(state));raise
 finally:stop.set();thread.join();status()

def make_request(settings,registry,registry_sha):
 if registry.get('status')!='complete' or set(registry['initializers'])!={str(s) for s in settings['seeds']}:raise ValueError('Initializer cohort mismatch')
 return dict(settings,initializers=registry['initializers'],initializers_registry_sha256=registry_sha)

def main(args):
 import torch
 if torch.cuda.is_initialized():raise RuntimeError('CPU stage must not initialize CUDA')
 seeds=parse_seeds(args.seeds);source=verify_sources();result=initializers(seeds,args.output,source)
 cache=Path(args.cache).absolute()/'CACHE_COMPLETE.json';training=Path(args.data).absolute()/'training.jsonl'
 settings=dict(seeds=list(seeds),cache_complete_path=str(cache),cache_complete_sha256=sha(cache),training_jsonl_path=str(training),training_jsonl_sha256=sha(training),output=str(Path(args.training_output).absolute()))
 request=make_request(settings,result,sha(Path(args.output)/'INITIALIZERS_COMPLETE.json'));p=Path(args.output)/'REQUEST.json';dump(p,request)
 from .prepare import prepare
 prepare(p,sha(p),Path(args.output)/'config')
 dump(Path(args.output)/'CPU_STAGE_COMPLETE.json',dict(status='complete',config_sha256=sha(Path(args.output)/'config/CONFIG.json'),source_manifest_sha256=source,seeds=list(seeds),cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--seeds',help='comma-separated subset of the eight registered seeds; default all')
 p.add_argument('--cache',required=True,help='build_cache output directory');p.add_argument('--data',required=True,help='build_passages output directory')
 p.add_argument('--output',required=True);p.add_argument('--training-output',required=True);main(p.parse_args())
