"""Exact small contracts and immutable authentication; no CUDA import."""
import hashlib,json,os,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SEEDS=(42,123,456,789,2026,2027,2028,2029)
OBJECTIVES=('standalone_memory','joint_base_anchored');READOUTS=('mlpmemory','residual')
GRID=(0.,.1,.2,.3,.4,.5,.75,1.)
NATIVE={'standalone_memory':'mlpmemory','joint_base_anchored':'residual'}
COUNTS={'nq':3610,'triviaqa':11313,'webq':2032,'hotpotqa':7405,'popqa':14267}
SUBSET={'nq':820,'triviaqa':819,'webq':819,'hotpotqa':819,'popqa':819}
DEV={'nq':512,'triviaqa':512}
RUNTIME=dict(base_dtype='bfloat16',memory_dtype='float32',attention='eager',use_cache=False,batch=16,input_cap=384,padding='batch_longest_left',max_new_tokens=12,tf32=False,gold='complete_no_EOS',gold_context_cap=384)
def encoded(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(encoded(x)).hexdigest()
def read(path):
 def unique(pairs):
  d={}
  for k,v in pairs:
   if k in d:raise ValueError('duplicate JSON key')
   d[k]=v
  return d
 def bad(x):raise ValueError('nonfinite JSON '+x)
 return json.loads(Path(path).read_bytes(),object_pairs_hook=unique,parse_constant=bad)
def exact(a,b):
 if type(a)!=type(b):raise ValueError('identity type mismatch')
 if isinstance(a,dict):
  if a.keys()!=b.keys():raise ValueError('identity keys mismatch')
  for k in a:exact(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  if len(a)!=len(b):raise ValueError('identity length mismatch')
  for x,y in zip(a,b):exact(x,y)
 elif a!=b:raise ValueError('identity value mismatch')
def physical(path):
 q=Path(path).absolute()
 if any(x.is_symlink() for x in (q,*q.parents)) or q!=q.resolve():raise ValueError('physical nonsymlink path required')
 return q
def sha(path):
 h=hashlib.sha256()
 with physical(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fresh(path,value):
 with Path(path).open('xb') as f:f.write(encoded(value)+b'\n')
def atomic(path,value):
 q=Path(path);tmp=q.with_name(q.name+'.tmp-'+str(os.getpid()))
 with tmp.open('xb') as f:f.write(encoded(value)+b'\n')
 os.replace(tmp,q)
def rows(path):
 with Path(path).open() as f:return [json.loads(x) for x in f]
def write_rows(path,rr):
 with Path(path).open('xb') as f:
  for r in rr:f.write(encoded(r)+b'\n')
def stat_record(path):
 s=physical(path).stat()
 if not stat.S_ISREG(s.st_mode):raise ValueError('regular file required')
 return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def record(path,expected=None):
 path=physical(path);before=stat_record(path);h=sha(path);exact(before,stat_record(path))
 if expected is not None and h!=expected:raise ValueError('payload SHA '+str(path))
 return dict(path=str(path),sha256=h,stat=before)
def verify(r):
 exact(stat_record(r['path']),r['stat'])
 if r['stat']['size']<=64*1024*1024 and sha(r['path'])!=r['sha256']:raise ValueError('small payload drift')
 exact(stat_record(r['path']),r['stat']);return Path(r['path'])
def source(expected=None):
 h=digest({q.name:sha(q) for q in sorted(ROOT.glob('*.py'))})
 if expected is not None and expected!=h:raise ValueError('source SHA')
 return h
def key(seed,objective,readout):return f'{seed}/{objective}/{readout}'
def cid(seed,objective):return f'{objective}-s{seed}'
def cohort_models(seeds):
 if not seeds or len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):raise ValueError('registered distinct seeds required')
 return {cid(s,o) for s in seeds for o in OBJECTIVES}
def config(path,expected=None,check_assets=True):
 h=sha(path)
 if expected is not None and h!=expected:raise ValueError('config hash')
 c=read(path);source(c['source_manifest_sha256']);exact(c['runtime'],RUNTIME)
 if c['schema']!='tec-evaluation-config-v1' or set(c['models'])!=cohort_models(c['seeds']):raise ValueError('config schema/cohort')
 if check_assets:
  verify_base(c)
  for v in c['population'].values():verify(v)
  for model in c['models'].values():
   for r in model['files'].values():verify(r)
 c['_sha256']=h;c['_path']=str(Path(path).absolute());return c

def verify_base(c):
 view=Path(c['base'])
 for name,r in c['base_files'].items():
  verify(r)
  if (view/name).resolve(strict=True)!=Path(r['path']):raise ValueError('Base loader file changed')
