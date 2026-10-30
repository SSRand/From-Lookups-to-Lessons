"""CPU-only exact per-tensor fresh initializer family, shared by both objectives."""
import argparse,random,gc
from pathlib import Path
from .protocol import *

def create(seed,output):
 import torch
 from safetensors.numpy import save_file
 if seed not in SEEDS:raise ValueError('Unregistered full seed')
 if torch.cuda.is_initialized():raise RuntimeError('Initializer must be CPU-only')
 out=Path(output).absolute()
 if out.resolve()!=out or out.exists():raise ValueError('Fresh physical initializer directory required')
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 out.mkdir(parents=True);cfg=read(HERE/'architecture.json');dump(out/'config.json',cfg)
 # Named tensor generation is independent of parameter iteration order and host.
 state={n:initial_tensor(n,s,seed) for n,s in tensor_shapes(8).items()}
 import hashlib
 tensor_hashes={n:hashlib.sha256(v.tobytes()).hexdigest() for n,v in state.items()}
 known=read(HERE/'ORIGINAL_INITIALIZERS.json')['tensor_sha256']
 if str(seed) in known and tensor_hashes!=known[str(seed)]:raise ValueError('Exact known original full-seed initializer differs')
 save_file(state,str(out/'model.safetensors'),metadata={'format':'pt'});del state;gc.collect()
 artifacts={n:sha(out/n) for n in ('model.safetensors','config.json')}
 dump(out/'COMPLETE.json',dict(schema='teacher-exposure-fresh-initializer-v1',status='complete',seed=seed,depth=8,parameters=parameter_count(8),source_manifest_sha256=verify_sources(),rule=scientific_protocol()['initialization'],tensor_sha256=tensor_hashes,artifacts=artifacts,cuda_initialized=torch.cuda.is_initialized()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();create(a.seed,a.output)
