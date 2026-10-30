"""CPU authentication of one completed training cohort; writes the evaluation CONFIG.

  python -m shared_teacher_ablation.eval.configure --training-root TRAIN --population-root POP --base BASE --output EVAL
writes EVAL/CONFIG.json and prints its path and SHA-256.
"""
import argparse,math
from pathlib import Path
from . import protocol as p
from . import population

def configure(a):
 train=p.physical(a.training_root);out=p.physical(a.output);out.mkdir(parents=True,exist_ok=True)
 target=out/'CONFIG.json'
 if target.exists():raise ValueError('fresh evaluation CONFIG required')
 done=p.read(train/'TRAINING_COMPLETE.json')
 if done['status']!='complete' or done['complete'] is not True or done['completed_training']!=done['total_training']:raise ValueError('complete training cohort required')
 seeds=[s for s in p.SEEDS if s in done['seeds']]
 if set(done['models'])!=p.cohort_models(seeds):raise ValueError('paired cohort models')
 models={}
 for cid in sorted(done['models']):
  root=train/'trained'/cid;r=p.read(root/'COMPLETE.json')
  if p.sha(root/'COMPLETE.json')!=done['models'][cid]['completion_sha256'] or r['status']!='complete':raise ValueError('model completion edge')
  if r['steps']!=32768 or r['positions_seen']!=8388608 or r['unique_cached_positions']!=1048576:raise ValueError('terminal dose')
  c=r['cell'];files={}
  for name in ('model.safetensors','config.json'):
   rel='milestones/32768/checkpoint/'+name;files[name]=p.record(root/rel,r['artifacts'][rel])
  from safetensors import safe_open
  with safe_open(files['model.safetensors']['path'],framework='pt',device='cpu') as f:
   shapes={k:tuple(f.get_slice(k).get_shape()) for k in f.keys()};dtypes={f.get_slice(k).get_dtype() for k in f.keys()}
  if sum(math.prod(x) for x in shapes.values())!=1543540736 or dtypes!={'F32'}:raise ValueError('exact8L FP32 checkpoint shape')
  models[cid]=dict(seed=c['seed'],objective=c['objective'],files=files,checkpoint=str(root/'milestones/32768/checkpoint'),training=r,training_completion_sha256=p.sha(root/'COMPLETE.json'))
 for s in seeds:
  a1=models[p.cid(s,p.OBJECTIVES[0])]['training'];a2=models[p.cid(s,p.OBJECTIVES[1])]['training']
  for k in ('initializer_sha256','initializer_config_sha256','order_sha256','cache_arrays_sha256','steps','positions_seen','unique_cached_positions','parameter_count'):p.exact(a1[k],a2[k])
 pop=p.physical(a.population_root);population.load_bundle(pop)
 poprecords={n:p.record(pop/n) for n in ('test.jsonl','dev.jsonl','diagnostic.jsonl','COMPLETE.json')}
 base=Path(a.base).absolute();basefiles={}
 for q in sorted(base.iterdir()):
  if q.is_file() and q.name!='consolidated.safetensors':basefiles[q.name]=p.record(q.resolve(strict=True))
 c=dict(schema='tec-evaluation-config-v1',cohort=a.cohort or 'seeds-'+'-'.join(map(str,seeds)),seeds=seeds,source_manifest_sha256=p.source(),runtime=p.RUNTIME,models=models,base=str(base),base_files=basefiles,population=poprecords,parent=dict(training_complete_sha256=p.sha(train/'TRAINING_COMPLETE.json'),source_manifest_sha256=done['source_manifest_sha256']),output=str(out))
 p.fresh(target,c);p.config(target,p.sha(target))
 print(p.encoded(dict(config=str(target),config_sha256=p.sha(target))).decode())
def main():
 ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 for n in ('training-root','population-root','base','output'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--cohort',help='label for this training cohort; default derived from its seeds')
 configure(ap.parse_args())
if __name__=='__main__':main()
