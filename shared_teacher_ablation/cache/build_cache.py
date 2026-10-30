"""Foreground multi-GPU cache producer for the shared BM25-RAG teacher.

For every training passage it stores, at the 32 supervised positions, the BF16
memory input h_t and Base logits of the student prefix, and the top-64
normalized teacher distribution of the same BF16 Base conditioned on the BM25
evidence prompt. Usage (from the repository root):
  python -m shared_teacher_ablation.cache.build_cache --base BASE --data DATA --output OUT [--gpus 0,1,...]
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .contract import array_specs,partition_rows,sha,validate_row,write_json

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
MODULE='shared_teacher_ablation.cache.build_cache'


def read(path):return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:return [json.loads(line) for line in f if line.strip()]


def source_identity():
    return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}


def runtime():
    import torch
    return {'python':sys.version,'executable':sys.executable,
            'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','safetensors','tokenizers')},
            'cuda':torch.version.cuda,'base_dtype':'bfloat16','attention':'eager','tf32':False}


def base_files(base):
    return sorted(p for p in Path(base).iterdir() if p.is_file() and p.name!='consolidated.safetensors')


def configure(a):
    out=Path(a.output).absolute();data=Path(a.data).absolute()
    if out.exists() and any(out.iterdir()):raise ValueError('Cache output must be new')
    gpus=a.gpus.split(',') if a.gpus else [x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x]
    if not gpus or len(set(gpus))!=len(gpus):raise ValueError('Distinct GPU devices required')
    training=rows(data/'training.jsonl')
    c={'base':str(Path(a.base).absolute()),'data_complete':str(data/'COMPLETE.json'),'data_complete_sha256':sha(data/'COMPLETE.json'),
       'training_jsonl':str(data/'training.jsonl'),'training_jsonl_sha256':sha(data/'training.jsonl'),'n_examples':len(training),
       'output':str(out),'gpus':gpus}
    out.mkdir(parents=True,exist_ok=True);write_json(out/'CONFIG.json',c)
    return c,out/'CONFIG.json'


def authenticate(c,config):
    actual_runtime=runtime()
    if sha(c['data_complete'])!=c['data_complete_sha256']:raise ValueError('Data completion identity differs')
    data_done=read(c['data_complete'])
    if data_done.get('status')!='complete' or data_done['artifacts']['training.jsonl']!=c['training_jsonl_sha256']:
        raise ValueError('Unaccepted training population')
    if sha(c['training_jsonl'])!=c['training_jsonl_sha256']:raise ValueError('Training identity differs')
    data=rows(c['training_jsonl'])
    if len(data)!=c['n_examples'] or len({r['stable_id'] for r in data})!=len(data):raise ValueError('Training population differs')
    for row in data:validate_row(row,32)
    files={}
    for p in base_files(c['base']):
        st=p.stat();files[p.name]={'sha256':sha(p),'stat':[st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]}
    out=Path(c['output'])
    if any(x.name!='CONFIG.json' for x in out.iterdir()):raise ValueError('Cache output must be new')
    identity={'config_sha256':sha(config),'source':source_identity(),'runtime':actual_runtime,
              'base_files':files,'training_jsonl_sha256':c['training_jsonl_sha256'],'epoch':time.time()}
    write_json(out/'AUTH.json',identity)
    return data,identity


def worker(c,config,rank,workers):
    """Compute BF16 memory inputs, Base logits, top-64 teacher distributions and gold ids
    for one shard of the supervised positions."""
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    auth=read(Path(c['output'])/'AUTH.json')
    if sha(config)!=auth['config_sha256'] or source_identity()!=auth['source']:raise ValueError('Worker source/config differs')
    for name,record in auth['base_files'].items():
        st=(Path(c['base'])/name).stat()
        if [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]!=record['stat']:raise ValueError('Base changed after authentication')
    if sha(c['training_jsonl'])!=auth['training_jsonl_sha256']:raise ValueError('Training rows changed')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=rows(c['training_jsonl']);indices=partition_rows(len(data),4,workers,rank)
    out=Path(c['output'])/'shards'/str(rank);out.mkdir(parents=True)
    np.save(out/'row_indices.npy',np.asarray(indices,dtype=np.int64))
    specs=array_specs(len(indices)*32)
    arrays={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,(shape,dtype) in specs.items()}
    tok=AutoTokenizer.from_pretrained(c['base'],local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(c['base'],local_files_only=True,torch_dtype=torch.bfloat16,
                                             attn_implementation='eager',low_cpu_mem_usage=True).cuda().eval()
    model.requires_grad_(False);capture={}
    if {p.dtype for p in model.parameters()}!={torch.bfloat16}:raise ValueError('Unexpected Base dtype')
    def hook(module,args,kwargs):capture['hidden']=args[0].detach()
    handle=model.model.layers[-1].mlp.register_forward_pre_hook(hook,with_kwargs=True)
    started=time.monotonic();offset=0
    with torch.inference_mode():
        for start in range(0,len(indices),4):
            batch_rows=[data[i] for i in indices[start:start+4]]
            for kind in ('student','teacher'):
                seqs=[r[kind+'_ids'] for r in batch_rows];width=max(map(len,seqs))
                ids=torch.full((len(seqs),width),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
                for j,seq in enumerate(seqs):ids[j,:len(seq)]=torch.tensor(seq,device='cuda');mask[j,:len(seq)]=1
                result=model(input_ids=ids,attention_mask=mask,use_cache=False)
                positions=torch.tensor([r[kind+'_positions'] for r in batch_rows],device='cuda')
                batch=torch.arange(len(seqs),device='cuda')[:,None]
                z=result.logits[batch,positions].reshape(-1,32768).float()
                take=z.shape[0]
                if not torch.isfinite(z).all():raise ValueError('Nonfinite Base/teacher logits')
                if kind=='student':
                    hidden=capture['hidden'][batch,positions].reshape(-1,4096)
                    if not torch.isfinite(hidden).all():raise ValueError('Nonfinite features')
                    arrays['hidden'][offset:offset+take]=hidden.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
                    arrays['base_logits'][offset:offset+take]=z.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
                else:
                    values,teacher_ids=z.topk(64,dim=-1)
                    arrays['teacher_ids'][offset:offset+take]=teacher_ids.int().cpu().numpy()
                    arrays['teacher_probs'][offset:offset+take]=values.softmax(-1).cpu().numpy()
                del result,z
            arrays['gold'][offset:offset+take]=np.asarray([r['target_ids'] for r in batch_rows]).reshape(-1)
            offset+=take
            if (start+len(batch_rows))%64==0:
                elapsed=time.monotonic()-started
                write_json(out/'STATUS.json',{'phase':'cache','epoch':time.time(),'rank':rank,'examples':start+len(batch_rows),
                    'positions':offset,'elapsed_seconds':elapsed,'projected_total_seconds':elapsed*len(indices)/(start+len(batch_rows))})
    handle.remove()
    if offset!=len(indices)*32:raise ValueError('Incomplete shard')
    for array in arrays.values():array.flush()
    files={name:{'relative_path':name+'.npy','sha256':sha(out/(name+'.npy')),'shape':list(shape),'dtype':dtype}
           for name,(shape,dtype) in specs.items()}
    write_json(out/'COMPLETE.json',{'status':'complete','rank':rank,'examples':len(indices),'positions':offset,
        'row_indices_sha256':sha(out/'row_indices.npy'),'arrays':files,'runtime':runtime(),
        'config_sha256':sha(config),'auth_sha256':sha(Path(c['output'])/'AUTH.json'),'seconds':time.monotonic()-started})



def merge(c,config,data,workers):
    import numpy as np
    out=Path(c['output']);n=len(data)*32;specs=array_specs(n)
    arrays={name:np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype=dtype,shape=shape) for name,(shape,dtype) in specs.items()}
    seen=set();shards={}
    for rank in range(workers):
        root=out/'shards'/str(rank);done=read(root/'COMPLETE.json')
        if done['status']!='complete' or done['config_sha256']!=sha(config):raise ValueError('Shard receipt mismatch')
        indices=np.load(root/'row_indices.npy');expected=partition_rows(len(data),4,workers,rank)
        if set(done['arrays'])!=set(specs):raise ValueError('Incomplete or extra shard arrays')
        if done.get('rank')!=rank or done.get('examples')!=len(expected) or done.get('positions')!=len(expected)*32:
            raise ValueError('Shard rank/population differs')
        if done.get('auth_sha256')!=sha(out/'AUTH.json'):raise ValueError('Shard producer authentication differs')
        if indices.tolist()!=expected or sha(root/'row_indices.npy')!=done['row_indices_sha256']:raise ValueError('Shard row identity differs')
        if seen.intersection(expected):raise ValueError('Duplicate shard rows')
        seen.update(expected)
        for name,record in done['arrays'].items():
            if record['relative_path']!=name+'.npy':raise ValueError('Unexpected shard array path')
            path=root/record['relative_path']
            if sha(path)!=record['sha256']:raise ValueError('Shard array hash differs')
            arr=np.load(path,mmap_mode='r');expected_shape,expected_dtype=array_specs(len(indices)*32)[name]
            if tuple(arr.shape)!=expected_shape or str(arr.dtype)!=expected_dtype:raise ValueError('Shard shape/dtype differs')
            # Consecutive original batches remain consecutive in each shard.
            for start in range(0,len(indices),4):
                src=arr[start*32:(start+4)*32];target=int(indices[start])*32
                if name in ('hidden','base_logits') and np.any((src & 0x7f80)==0x7f80):raise ValueError('Nonfinite BF16 cache')
                if name=='teacher_probs' and (not np.isfinite(src).all() or (src<0).any() or not np.allclose(src.sum(-1),1,atol=1e-5)):
                    raise ValueError('Teacher distribution invalid')
                if name in ('teacher_ids','gold') and ((src<0).any() or (src>=32768).any()):raise ValueError('Invalid token ID')
                if name=='gold':
                    expected_gold=np.asarray([data[int(j)]['target_ids'] for j in indices[start:start+4]],dtype=np.int64).reshape(-1)
                    if not np.array_equal(src,expected_gold):raise ValueError('Gold does not match original causal targets')
                arrays[name][target:target+128]=src
        shards[str(rank)]=sha(root/'COMPLETE.json')
    if seen!=set(range(len(data))):raise ValueError('Incomplete merged population')
    for a in arrays.values():a.flush()
    files={name:{'path':name+'.npy','relative_path':name+'.npy','sha256':sha(out/(name+'.npy')),'shape':list(shape),'dtype':dtype}
           for name,(shape,dtype) in specs.items()}
    receipt={'schema':'shared-rag-cache-v1','status':'complete','n_positions':n,'n_examples':len(data),'target_tokens':32,
        'training_jsonl_sha256':c['training_jsonl_sha256'],'data_complete_sha256':c['data_complete_sha256'],
        'base_files':{k:v['sha256'] for k,v in read(out/'AUTH.json')['base_files'].items()},'arrays':files,'runtime':runtime(),
        'config_sha256':sha(config),'source':source_identity(),'shards':shards,'epoch':time.time(),
        'storage':'uint16 bit-preserving BF16 features/logits; normalized FP32 top64 teacher probabilities'}
    write_json(out/'CACHE_COMPLETE.json',receipt)
    write_json(out/'STATUS.json',{'phase':'complete','epoch':time.time(),'n_positions':n,'manifest_sha256':sha(out/'CACHE_COMPLETE.json')})


def pipeline(c,config):
    devices=c['gpus']
    data,auth=authenticate(c,config);out=Path(c['output']);children=[];handles=[]
    def stop(signum,frame):raise KeyboardInterrupt('Termination requested')
    signal.signal(signal.SIGTERM,stop)
    try:
        for rank,device in enumerate(devices):
            log=(out/('worker-'+str(rank)+'.log')).open('xb');handles.append(log)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            child=subprocess.Popen([sys.executable,'-B','-m',MODULE,'--config',str(config),'--worker',str(rank)],
                                   env=env,cwd=str(REPO),stdout=log,stderr=subprocess.STDOUT)
            children.append(child)
        while any(p.poll() is None for p in children):
            bad=[(i,p.returncode) for i,p in enumerate(children) if p.poll() not in (None,0)]
            if bad:raise RuntimeError('Cache worker failed: '+repr(bad))
            write_json(out/'STATUS.json',{'phase':'cache_workers','epoch':time.time(),
                'workers':{str(i):{'pid':p.pid,'returncode':p.poll()} for i,p in enumerate(children)}})
            time.sleep(10)
        if any(p.returncode for p in children):raise RuntimeError('Cache worker failed')
        write_json(out/'STATUS.json',{'phase':'merging','epoch':time.time()})
        merge(c,config,data,len(devices))
    except BaseException as exc:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        write_json(out/'STATUS.json',{'phase':'failed','epoch':time.time(),'error':repr(exc)})
        raise
    finally:
        for h in handles:h.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base',help='local Mistral-7B-v0.3 directory');parser.add_argument('--data',help='build_passages output directory')
    parser.add_argument('--output');parser.add_argument('--gpus',help='comma-separated devices; default CUDA_VISIBLE_DEVICES')
    parser.add_argument('--config',help=argparse.SUPPRESS);parser.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker is None:
        c,config=configure(args);pipeline(c,config.resolve())
    else:
        config=Path(args.config).resolve();c=read(config);worker(c,config,args.worker,len(c['gpus']))
