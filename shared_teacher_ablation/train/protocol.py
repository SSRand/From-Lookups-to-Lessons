"""Frozen scientific scope; no quality-dependent selection or schedule changes."""
import hashlib,json
from pathlib import Path
import numpy as np
from .contract import sha,canonical,read,rows,dump,tensor_shapes,initial_tensor,parameter_count,validate_cache_arrays
HERE=Path(__file__).resolve().parent
SEEDS=(42,123,456,789,2026,2027,2028,2029)
OBJECTIVES=('standalone_memory','joint_base_anchored')
N=1048576;EPOCHS=8;BATCH=256;MICRO=32;STEPS=32768;ALPHA=.5
MILESTONE_EPOCHS=(1,2,4,6,8)

def parse_seeds(value):
 seeds=tuple(int(x) for x in str(value).split(',') if x.strip()) if value else SEEDS
 if not seeds or len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):raise ValueError('Registered distinct seeds required')
 return tuple(s for s in SEEDS if s in seeds)

def cells(seeds):
 if not seeds or len(set(seeds))!=len(seeds) or any(s not in SEEDS for s in seeds):raise ValueError('Registered distinct seeds required')
 return [dict(id=f'{kind}-s{seed}',seed=seed,objective=kind,depth=8) for seed in SEEDS if seed in seeds for kind in OBJECTIVES]

def orders(n,epochs,seed):
 if any(type(v) is not int or v<=0 for v in (n,epochs,seed)):raise ValueError('Positive integer geometry required')
 values=[np.random.default_rng(seed+e).permutation(n).astype('<i8') for e in range(epochs)]
 h=hashlib.sha256()
 for x in values:h.update(x.tobytes())
 return values,h.hexdigest()

def learning_rate(step,total=STEPS):
 if type(step) is not int or type(total) is not int or total<=32 or not 0<=step<total:raise ValueError('LR step outside fixed schedule')
 if step<32:return .0005*(step+1)/32
 if step==total-1:return .00005
 return .0005+(.00005-.0005)*(step-31)/(total-32)

def probe_indices(n=N):
 if n%32 or n<4096:raise ValueError('Whole passage cache required')
 ordinal=sorted(range(n//32),key=lambda i:hashlib.sha256(f'teacher-exposure-probe-v1|{i}'.encode()).digest())[:128]
 return [i*32+j for i in ordinal for j in range(32)]

def scientific_protocol():
 return dict(schema='same-teacher-exposure-training-protocol-v1',seeds=list(SEEDS),objectives=list(OBJECTIVES),depth=8,n_examples=N//32,positions=N,epochs=EPOCHS,steps=STEPS,batch=BATCH,micro=MICRO,alpha_ce=ALPHA,teacher='same sparse top64 RAG conditional q, normalized support',loss='(1-alpha)*sum(q*(log(q)-logsoftmax(logits)[support])) + alpha*gold_CE, mean over positions',objective_logits={'standalone_memory':'memory','joint_base_anchored':'base + memory'},precision='BF16 cached hidden/base -> FP32 memory/AdamW',optimizer={'name':'torch.optim.AdamW','weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'clip_grad_norm':1.},schedule={'warmup_steps':32,'peak_lr':.0005,'final_lr':.00005,'after_warmup':'linear; last update exactly final_lr'},milestone_epochs=list(MILESTONE_EPOCHS),initialization='per-tensor PCG64 Normal(0,.02) seeded by sha256(p1b-fresh-v1|seed|name); norm ones',selection='fixed epoch8 primary, retained epoch4 secondary; no quality-dependent stopping',diagnostics='fixed128 training-passage probe at epoch0/1/2/4/6/8; no unseen generalization or stationarity acceptance claim')

def verify_sources(expected=None):
 h=canonical({p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json')})
 if expected is not None and h!=expected:raise ValueError('Scientific source drift')
 return h
