"""CPU scientific contract: fresh-random initializer and cache array checks."""
import hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
DEPTHS=(2,4,8)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open() as f:return [json.loads(x) for x in f if x.strip()]
def dump(path,value):
    """Atomically publish immutable JSON; identical concurrent writers are safe."""
    path=Path(path);text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
            temporary=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
        try:os.link(temporary,path)
        except FileExistsError:
            if path.read_text()!=text:raise ValueError('Immutable receipt differs: '+str(path))
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
def parameter_count(depth,hidden=4096,intermediate=14336,vocab=32768):
    if depth not in DEPTHS:raise ValueError('Unexpected depth')
    return depth*(3*hidden*intermediate+hidden)+hidden+vocab*hidden
def tensor_shapes(depth,hidden=4096,intermediate=14336,vocab=32768):
    parameter_count(depth,hidden,intermediate,vocab)
    spec={'norm.weight':(hidden,),'lm_head.weight':(vocab,hidden)}
    for i in range(depth):
        p=f'layers.{i}.';spec[p+'input_layernorm.weight']=(hidden,)
        for name,shape in [('gate_proj',(intermediate,hidden)),('up_proj',(intermediate,hidden)),('down_proj',(hidden,intermediate))]:spec[p+'mlp.'+name+'.weight']=shape
    return dict(sorted(spec.items()))
def initial_tensor(name,shape,seed=42):
    if name.endswith('norm.weight'):return np.ones(shape,dtype=np.float32)
    number=int.from_bytes(hashlib.sha256(f'p1b-fresh-v1|{seed}|{name}'.encode()).digest()[:16],'little')
    return np.random.Generator(np.random.PCG64(number)).normal(0,.02,size=shape).astype('<f4')
def validate_cache_arrays(arrays,training,hidden=4096,vocab=32768,topk=64):
    gold=[]
    for row in training:
        target=row['target_ids'];gold.extend(target)
        for kind in ['student','teacher']:
            pos=row[kind+'_positions'];seq=row[kind+'_ids']
            if len(pos)!=len(target) or pos!=list(range(pos[0],pos[0]+len(target))) or any(p<0 or p+1>=len(seq) for p in pos):raise ValueError('Invalid shifted target geometry')
            if [seq[p+1] for p in pos]!=target:raise ValueError('Causal gold mismatch')
    n=len(gold)
    spec={'hidden':((n,hidden),'uint16'),'base_logits':((n,vocab),'uint16'),'teacher_ids':((n,topk),'int32'),'teacher_probs':((n,topk),'float32'),'gold':((n,),'int64')}
    if set(arrays)!=set(spec):raise ValueError('Cache array coverage')
    for k,(shape,dtype) in spec.items():
        if arrays[k].shape!=shape or arrays[k].dtype!=np.dtype(dtype):raise ValueError('Cache shape/dtype: '+k)
    if not np.array_equal(arrays['gold'],gold):raise ValueError('Cache gold order')
    for start in range(0,n,1024):
        ids=arrays['teacher_ids'][start:start+1024];p=arrays['teacher_probs'][start:start+1024]
        if np.any(ids<0) or np.any(ids>=vocab) or np.any(np.diff(np.sort(ids,axis=1),axis=1)==0):raise ValueError('Invalid teacher support')
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,rtol=0,atol=2e-6):raise ValueError('Invalid teacher probabilities')
        for name in ['hidden','base_logits']:
            bits=arrays[name][start:start+1024].astype(np.uint32)<<16
            if not np.isfinite(bits.view(np.float32)).all():raise ValueError('Nonfinite BF16 cache')
