"""Single full CPU authentication followed by immutable reuse."""
import stat,platform,importlib.metadata
from pathlib import Path
import numpy as np
from .protocol import *

def physical_child(root,relative):
 root=Path(root).absolute();rel=Path(relative)
 if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Relative contained path required')
 p=root/rel
 if p.resolve()!=p or not p.is_relative_to(root):raise ValueError('Symlink/path escape')
 return p

def snapshot(paths):
 result={}
 for value in paths:
  p=Path(value).absolute()
  if p.resolve()!=p:raise ValueError('Physical nonsymlink file required')
  s=p.lstat()
  if not stat.S_ISREG(s.st_mode):raise ValueError('Regular file required')
  rec={k:getattr(s,'st_'+k) for k in ('dev','ino','size','mtime_ns','ctime_ns')}
  if s.st_size<=32*1024**2:rec['small_sha256']=sha(p)
  result[str(p)]=rec
 return result

def verify_snapshot(paths,proof):
 if snapshot(paths)!=proof:raise ValueError('Authenticated immutable payload drift')

def validate_arrays(arrays,training,**kw):return validate_cache_arrays(arrays,training,**kw)

def software():
 return dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ('torch','numpy','transformers','safetensors','tokenizers')})

def cache_files(path,expected,training_path,training_sha):
 p=Path(path).absolute()
 if p.resolve()!=p or sha(p)!=expected:raise ValueError('Cache completion identity')
 r=read(p)
 for k,w in dict(status='complete',n_examples=N//32,n_positions=N,training_jsonl_sha256=training_sha).items():
  if type(r.get(k)) is not type(w) or r[k]!=w:raise ValueError('Cache contract: '+k)
 if not isinstance(r.get('data_complete_sha256'),str) or len(r['data_complete_sha256'])!=64:raise ValueError('Data producer identity absent')
 names={'hidden','base_logits','teacher_ids','teacher_probs','gold'}
 if set(r['arrays'])!=names:raise ValueError('Cache array map coverage')
 files={k:physical_child(p.parent,r['arrays'][k]['path']) for k in names}
 if len(set(files.values()))!=5:raise ValueError('Cache payload aliases')
 if sha(training_path)!=training_sha:raise ValueError('Training source changed')
 return r,files

def open_config(path,expected):
 p=Path(path).absolute()
 if p.resolve()!=p or sha(p)!=expected:raise ValueError('Config hash/path mismatch')
 c=read(p)
 if c.get('schema')!='teacher-exposure-host-config-v1' or c.get('status')!='complete' or c['source_manifest_sha256']!=verify_sources() or c['protocol']!=scientific_protocol():raise ValueError('Config/source/protocol mismatch')
 cells(c['assigned_seeds'])
 if set(c['initializers'])!={str(x) for x in c['assigned_seeds']}:raise ValueError('Assigned paired seeds mismatch')
 verify_snapshot(list(c['immutable_stats']),c['immutable_stats'])
 return c

def open_arrays(config):return {k:np.load(v,mmap_mode='r',allow_pickle=False) for k,v in config['cache_arrays'].items()}

def model_architecture(config):
 wanted=read(HERE/'architecture.json')
 for k in ('hidden_size','intermediate_size','vocab_size','num_hidden_layers','hidden_act','rms_norm_eps'):
  if type(config.get(k)) is not type(wanted[k]) or config[k]!=wanted[k]:raise ValueError('Initializer architecture: '+k)

def validate_initializer(directory,seed):
 from safetensors import safe_open
 import hashlib
 p=Path(directory).absolute();model_architecture(read(p/'config.json'));hashes={}
 with safe_open(str(p/'model.safetensors'),framework='np') as f:
  shapes=tensor_shapes(8)
  if set(f.keys())!=set(shapes):raise ValueError('Initializer tensor coverage')
  for name,shape in shapes.items():
   v=f.get_tensor(name)
   if v.shape!=shape or v.dtype!=np.dtype('float32') or not np.isfinite(v).all():raise ValueError('Initializer shape/precision/finite')
   h=hashlib.sha256(v.tobytes()).hexdigest();expected=hashlib.sha256(initial_tensor(name,shape,seed).tobytes()).hexdigest()
   if h!=expected:raise ValueError('Original fresh-seed rule differs: '+name)
   hashes[name]=h
 known=read(HERE/'ORIGINAL_INITIALIZERS.json')['tensor_sha256']
 if str(seed) in known and hashes!=known[str(seed)]:raise ValueError('Original accepted fresh-seed tensor identity differs')
 return hashes
