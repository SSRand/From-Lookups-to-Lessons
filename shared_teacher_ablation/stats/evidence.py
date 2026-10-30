"""Byte-bound complete inputs; reuse the evaluation identity/scorer/seal checks."""
import hashlib,json,math,stat
from pathlib import Path,PurePosixPath
import numpy as np
from ..eval import protocol,population,barrier,analyze

CONTRAST='joint_base_anchored-minus-standalone_memory'
TOLERANCE=2e-11

def physical(path):
 q=Path(path).absolute()
 if q!=q.resolve() or any(x.is_symlink() for x in (q,*q.parents)):raise ValueError('physical path required')
 return q
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def safe_file(path,expected=None):
 q=physical(path);before=q.stat()
 if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or before.st_size>1024**3:raise ValueError('bounded regular non-hardlink file required')
 if expected is not None:
  if type(expected)is not str or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or digest(q)!=expected:raise ValueError('actual file SHA mismatch '+str(q))
 if q.stat()!=before:raise ValueError('input changed during authentication')
 return q
def read(path):
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('duplicate JSON key')
   out[k]=v
  return out
 def bad(s):raise ValueError('nonfinite JSON '+s)
 q=safe_file(path);before=q.stat();raw=q.read_bytes()
 if q.stat()!=before:raise ValueError('input changed during JSON read')
 return json.loads(raw,object_pairs_hook=unique,parse_constant=bad)
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def exact(a,b):
 if type(a)is not type(b):raise ValueError('typed identity')
 if isinstance(b,dict):
  if set(a)!=set(b):raise ValueError('identity fields')
  for k in b:exact(a[k],b[k])
 elif isinstance(b,(list,tuple)):
  if len(a)!=len(b):raise ValueError('identity lengths')
  for x,y in zip(a,b):exact(x,y)
 elif a!=b:raise ValueError('identity value')
def close(a,b):
 if isinstance(b,dict):
  if type(a)is not dict or set(a)!=set(b):raise ValueError('numeric fields')
  for k in b:close(a[k],b[k])
 elif isinstance(b,list):
  if type(a)is not list or len(a)!=len(b):raise ValueError('numeric lengths')
  for x,y in zip(a,b):close(x,y)
 elif type(b)in (int,float):
  if type(a)not in (int,float) or not math.isfinite(a) or not math.isfinite(b) or abs(a-b)>TOLERANCE*max(1,abs(b)):raise ValueError('numeric mismatch')
 else:exact(a,b)
def relative(s):
 if type(s)is not str or not s or '\\' in s or PurePosixPath(s).is_absolute() or any(x in ('','.','..') for x in s.split('/')):raise ValueError('safe relative artifact required')
 return s
def expected_files(models,phase):
 if phase not in ('calibration','test'):raise ValueError('known full phase required')
 files={'CONFIG.json','WORKERS_COMPLETE.json'}
 roles=('mlpmemory','residual') if phase=='calibration' else ('native_selected','native_selected_diagnostic','cross_selected','fixed1-mlpmemory','fixed1-residual')
 for cid in models:
  for role in roles:
   files.update(f'models/{cid}/{role}/{n}' for n in ('predictions.jsonl','COMPLETE.json'))
 if phase=='test':
  files.add('GLOBAL_BARRIER.json')
  for rank in range(len(models)):files.update(f'base/{rank}/{n}' for n in ('predictions.jsonl','COMPLETE.json'))
 return files
def closure(root,phase,expected_sha,p,barrier_sha,pop_sha,train_source):
 root=physical(root);name='COMPLETION.json' if phase=='test' else 'CALIBRATION_COMPLETE.json';safe_file(root/name,expected_sha);done=read(root/name);c=read(root/'CONFIG.json');cohort=c['cohort'];n=len(c['models'])
 wanted=dict(schema='tec-evaluation-phase-v1',status='complete',complete=True,phase=phase,cohort=cohort,seeds=c['seeds'],source_manifest_sha256=p.source(),config_sha256=digest(root/'CONFIG.json'),training_complete_sha256=c['parent']['training_complete_sha256'],population_sha256=pop_sha,barrier_sha256=barrier_sha if phase=='test' else None,completed_models=n,policies=32 if phase=='test' else 2*n,scientific_acceptance=False)
 ntest=sum(p.COUNTS.values());ndiag=sum(p.SUBSET.values());ndev=sum(p.DEV.values());wanted['rows']=n*(ntest+4*ndiag)+ntest if phase=='test' else n*2*ndev*len(p.GRID)
 for k,v in wanted.items():exact(done[k],v)
 exact(c['source_manifest_sha256'],p.source());exact(c['runtime'],p.RUNTIME);exact(c['parent']['source_manifest_sha256'],train_source)
 exact(c['population']['COMPLETE.json']['sha256'],pop_sha)
 if set(c['models'])!=p.cohort_models(c['seeds']):raise ValueError('full paired cohort required')
 for cid,m in c['models'].items():
  if type(m['seed'])is not int or p.cid(m['seed'],m['objective'])!=cid:raise ValueError('cell identity')
  t=m['training']
  for k,v in dict(source_manifest_sha256=train_source,steps=32768,positions_seen=8388608,unique_cached_positions=1048576,parameter_count=1543540736,status='complete').items():exact(t[k],v)
  for name in ('model.safetensors','config.json'):exact(m['files'][name]['sha256'],t['artifacts']['milestones/32768/checkpoint/'+name])
 if set(done['artifacts'])!=expected_files(c['models'],phase):raise ValueError('exact phase artifact map')
 size=0
 for rel,h in done['artifacts'].items():
  q=safe_file(root/relative(rel),h);size+=q.stat().st_size
 if size>8*1024**3:raise ValueError('bounded cohort closure size')
 if phase=='test':safe_file(root/'GLOBAL_BARRIER.json',barrier_sha)
 w=read(root/'WORKERS_COMPLETE.json');exact(w['returncodes'],[0]*n)
 if type(w['commands'])is not list or len(w['commands'])!=n:raise ValueError('actual worker commands')
 for rank,cid in enumerate(sorted(c['models'])):
  cmd=w['commands'][rank]
  if len(cmd)!=(18 if phase=='test' else 14) or cmd[3]!='shared_teacher_ablation.eval.worker' or cmd[6:14]!=['--expected-config-sha256',digest(root/'CONFIG.json'),'--cell',cid,'--rank',str(rank),'--phase',phase]:raise ValueError('exact completed worker argv')
  if phase=='test' and (cmd[14]!='--barrier' or not Path(cmd[15]).is_absolute() or cmd[16:]!=['--expected-barrier-sha256',barrier_sha]):raise ValueError('worker actual seal binding')
 return c,done

def load_inputs(a,out):
 p=protocol
 bpath=safe_file(a.barrier);barrier_sha=digest(bpath);b=read(bpath);pop_sha=digest(Path(a.population_root)/'COMPLETE.json');train_source=b['training_source_manifest_sha256']
 configs={};completions=[]
 for root in a.snapshot:
  cohort=read(Path(root)/'COMPLETION.json')['cohort']
  if cohort in configs or cohort not in b['cohorts']:raise ValueError('duplicate or unknown cohort')
  c,done=closure(root,'test',None,p,barrier_sha,pop_sha,train_source);configs[cohort]=c;completions.append(dict(cohort=cohort,completion_sha256=digest(Path(root)/'COMPLETION.json')))
 seen=set()
 for root in a.calibration:
  cohort=read(Path(root)/'CALIBRATION_COMPLETE.json')['cohort']
  if cohort in seen or cohort not in configs:raise ValueError('duplicate/wrong calibration cohort')
  c,done=closure(root,'calibration',b['cohorts'][cohort]['calibration_complete_sha256'],p,barrier_sha,pop_sha,train_source);seen.add(cohort);exact(c,configs[cohort])
 if set(configs)!=set(b['cohorts']) or seen!=set(configs):raise ValueError('every sealed cohort test and calibration closure required')
 analysis_sha=digest(a.analysis);analysis=read(safe_file(a.analysis))
 for k,v in dict(population_sha256=pop_sha,barrier_sha256=barrier_sha,primary_population=38627,full_seed_count=8,primary_metric='equal_dataset_macro_F1_RES_minus_MLP',policies=b['policies']).items():exact(analysis[k],v)
 # Only now enter the scorer/population readers: all closures were complete.
 pop,_=population.load_bundle(a.population_root,pop_sha)
 seal=barrier.open_seal(a.barrier,barrier_sha)
 rebuilt=out/'REBUILT_GLOBAL_BARRIER.json';barrier.seal(a.calibration,a.population_root,pop_sha,rebuilt)
 if rebuilt.read_bytes()!=bpath.read_bytes():raise ValueError('all32 raw calibration policies must reproduce the seal bytes')
 cells={};base=None
 for root in a.snapshot:
  cc,bb,done=analyze.audit_host(root,pop,seal,barrier_sha)
  if set(cells)&set(cc):raise ValueError('duplicate fitted cell')
  cells.update(cc)
  if base is not None:
   for x,y in zip(base,bb,strict=True):
    for k in ('dataset','stable_id','input_sha256','gold_mask_sha256','prediction','token_nll','em','f1'):exact(x[k],y[k])
  base=bb
 if set(cells)!=p.cohort_models(p.SEEDS):raise ValueError('all eight paired runs required')
 matrices={o:np.asarray([[r['f1'] for r in cells[p.cid(s,o)]['native']] for s in p.SEEDS],dtype=np.float64) for o in p.OBJECTIVES}
 inputs=dict(evaluation_source_manifest_sha256=p.source(),cohort_completions=completions,barrier_sha256=barrier_sha,population_sha256=pop_sha,analysis_sha256=analysis_sha,closure_scope='Complete all16 model/test and all32 calibration closures; evaluation identity/scorer and exact regenerated seal')
 return pop['test'],matrices[p.OBJECTIVES[1]],matrices[p.OBJECTIVES[0]],analysis,inputs

def match_primary(result,analysis):
 close(result['point'],analysis['primary_contrast_points'][CONTRAST]['token_f1']);close(result['paired_run_differences'],analysis['paired_seed_F1_differences']);close(result['paired_run_sample_sd'],analysis['paired_seed_sample_sd'])
 q=result['intervals']['question'];old=analysis['primary_CIs'][CONTRAST]['macro']['token_f1']
 if q['status']=='available':close({k:q[k] for k in ('low','high')},old)
 return dict(point_matches=True,all_eight_paired_differences_match=True,original_question_CI=old,question_CI_matches=True if q['status']=='available' else None,question_CI_unavailable_reason=None if q['status']=='available' else q['status'],tolerance=TOLERANCE)
