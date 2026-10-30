"""Exact companion identities and completed-test closure; no GPU/model imports."""
import stat
from pathlib import Path
from ..eval import protocol as p
from ..eval import population,outputs,barrier,numerics
ROOT=Path(__file__).resolve().parent
ROLES=('native_selected','native_selected_diagnostic','cross_selected','fixed1-mlpmemory','fixed1-residual')

def source(expected=None):
 h=p.digest(dict(evaluation=p.source(),corrections={q.name:p.sha(q) for q in sorted(ROOT.glob('*.py'))}))
 if expected is not None:p.exact(h,expected)
 return h

def batch_plan(rr,shards=8):
 result=[]
 for i,b in enumerate(outputs.batches(rr)):
  x=dict(global_batch_id=i,rank=i%shards,dataset=b[0]['dataset'],B=len(b),generation_L=max(len(r['prompt_ids']) for r in b),gold_L=max(len(r['gold_input_ids']) for r in b),stable_ids=[r['stable_id'] for r in b],input_sha256s=[r['input_sha256'] for r in b],gold_mask_sha256s=[r['gold_mask_sha256'] for r in b],prompt_ids_sha256=p.digest([r['prompt_ids'] for r in b]),gold_input_ids_sha256=p.digest([r['gold_input_ids'] for r in b]))
  result.append(dict(x,geometry_sha256=p.digest(x)))
 return result

def original_artifacts(c):
 files={'CONFIG.json','GLOBAL_BARRIER.json','WORKERS_COMPLETE.json'}
 files.update(f'base/{r}/{n}' for r in range(len(c['models'])) for n in ('COMPLETE.json','predictions.jsonl'))
 files.update(f'models/{cid}/{role}/{n}' for cid in c['models'] for role in ROLES for n in ('COMPLETE.json','predictions.jsonl'))
 return files

def receipt_fields(c,bs):
 n=len(c['models']);ntest=sum(p.COUNTS.values());ndiag=sum(p.SUBSET.values())
 return dict(schema='tec-evaluation-phase-v1',status='complete',complete=True,phase='test',cohort=c['cohort'],seeds=c['seeds'],source_manifest_sha256=p.source(),config_sha256=c['_sha256'],training_complete_sha256=c['parent']['training_complete_sha256'],population_sha256=c['population']['COMPLETE.json']['sha256'],barrier_sha256=bs,completed_models=n,policies=32,rows=n*(ntest+4*ndiag)+ntest,scientific_acceptance=False)
def check_receipt_fields(d,c,bs):
 for k,v in receipt_fields(c,bs).items():p.exact(d[k],v)

def artifact_records(root,actual,expected):
 root=p.physical(root)
 if set(actual)!=set(expected):raise ValueError('exact artifact allowlist')
 records={};total=0
 for rel,h in actual.items():
  q=Path(rel)
  if q.is_absolute() or '..' in q.parts:raise ValueError('artifact path escape')
  q=p.physical(root/q)
  if not q.is_relative_to(root) or not stat.S_ISREG(q.stat().st_mode) or q.stat().st_nlink!=1:raise ValueError('regular nonlinked artifact')
  total+=q.stat().st_size
  if q.stat().st_size>512*1024**2 or total>8*1024**3:raise ValueError('bounded small closure')
  records[rel]=p.record(q,h)
 return records

def original_complete(c,bs,root=None):
 root=p.physical(root or Path(c['output'])/'test');d=p.read(root/'COMPLETION.json');check_receipt_fields(d,c,bs)
 records=artifact_records(root,d['artifacts'],original_artifacts(c))
 p.exact(p.sha(root/'CONFIG.json'),c['_sha256']);p.exact(p.sha(root/'GLOBAL_BARRIER.json'),bs)
 w=p.read(root/'WORKERS_COMPLETE.json');n=len(c['models']);p.exact(w['returncodes'],[0]*n)
 if len(w['commands'])!=n:raise ValueError('all original worker commands')
 for rank,cid in enumerate(sorted(c['models'])):
  a=w['commands'][rank]
  if len(a)!=18 or a[3]!='shared_teacher_ablation.eval.worker' or a[6:14]!=['--expected-config-sha256',c['_sha256'],'--cell',cid,'--rank',str(rank),'--phase','test'] or a[16:]!=['--expected-barrier-sha256',bs]:raise ValueError('original worker argv')
 return d,records,w

def base_identity(c,bs,original_sha,rank,companion_source):
 return dict(schema='tec-diagnostic-base-identity-v1',role='matched_diagnostic_base',cohort=c['cohort'],rank=rank,source_manifest_sha256=companion_source,original_source_manifest_sha256=p.source(),config_sha256=c['_sha256'],barrier_sha256=bs,original_complete_sha256=original_sha,population_sha256=c['population']['COMPLETE.json']['sha256'],base_files={n:v['sha256'] for n,v in c['base_files'].items()},runtime=c['runtime'],lambda_=0.)

def validate_rows(actual,rr,plan,rank,identity):
 expected=[(src,geometry) for b,geometry in zip(outputs.batches(rr),plan,strict=True) if geometry['rank']==rank for src in b]
 if len(actual)!=len(expected):raise ValueError('complete diagnostic rank coverage')
 for r,(src,g) in zip(actual,expected,strict=True):
  outputs.validate_row(r,src,identity,False)
  p.exact(r['global_batch_id'],g['global_batch_id']);p.exact(r['geometry_sha256'],g['geometry_sha256'])
 return actual

def shard(root,c,rr,plan,rank,bs,orig,src):
 root=Path(root)/'base'/str(rank);d=p.read(root/'COMPLETE.json');ident=base_identity(c,bs,orig,rank,src)
 for k,w in dict(schema='tec-diagnostic-base-shard-v1',status='complete',complete=True,rank=rank,identity=ident,rows=sum(len(x['stable_ids']) for x in plan if x['rank']==rank),batches=sum(x['rank']==rank for x in plan)).items():p.exact(d[k],w)
 artifact_records(root,d['artifacts'],{'predictions.jsonl'})
 return validate_rows(p.rows(root/'predictions.jsonl'),rr,plan,rank,ident),d

def companion_artifacts(shards):
 return {'CONFIG.json','GLOBAL_BARRIER.json','TEST_AUTH.json','BATCH_PLAN.json','WORKERS_COMPLETE.json'}|{f'base/{r}/{n}' for r in range(shards) for n in ('COMPLETE.json','predictions.jsonl')}
