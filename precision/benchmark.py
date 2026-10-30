"""Run one arm/precision of the deployment timing benchmark in a fresh CUDA process.

Expose exactly one GPU. Each arm runs in its own process, in the fixed order
Base BF16; Memory Decoder BF16, FP32; MLP Memory FP32; ResMem FP32; MLP Memory
BF16; ResMem BF16. Every condition (384 input tokens, batch 1 or 16, exactly 32
generated tokens, SDPA with KV cache) runs one first request, 3 warmups and 10
measured steady requests. Coefficients: lambda=.25 (MLP Memory,
Memory Decoder), gamma=.4 (ResMem).

    python -m precision.benchmark --arm base --out runs/precision/timing/base.bfloat16
    python -m precision.benchmark --arm res --memory-dtype float32 --memory RESMEM_CKPT --out runs/precision/timing/res.float32
    python -m precision.benchmark --arm md --memory-dtype bfloat16 --memory MD_CKPT --out runs/precision/timing/md.bfloat16
    python -m precision.summarize_timing runs/precision/timing
"""
import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import time

PROCESS_STARTED=time.perf_counter()


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm',required=True,choices=['base','mlp','res','md'])
    parser.add_argument('--memory-dtype',default='bfloat16',choices=['bfloat16','float32'])
    parser.add_argument('--base',default='mistralai/Mistral-7B-v0.3')
    parser.add_argument('--memory',help='memory checkpoint (MLP Memory, ResMem or Memory Decoder)')
    parser.add_argument('--prompts',type=Path,default=None,help='profiling prompts (default: precision/timing_prompts.json)')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--warmups',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    if args.arm!='base' and not args.memory:raise ValueError('--memory is required for memory arms')
    if args.warmups < 1 or args.repeats < 1:raise ValueError('warmups/repeats must be positive')
    if not args.smoke and (args.warmups < 3 or args.repeats < 10):
        raise ValueError('production protocol requires at least 3 warmups and 10 repeats')
    args.out.mkdir(parents=True)
    import torch
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    from resmem.memory import MistralMLPModel
    from precision.runtime import Runtime,generate,parameter_inventory
    from precision.metrics import aggregate
    from precision.timing_inputs import PROMPTS,freeze
    torch.set_num_threads(2)
    torch.manual_seed(20260920)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU for comparable measurements')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    import_and_cuda_setup_seconds=time.perf_counter()-PROCESS_STARTED
    started=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(args.base)
    tokenizer.padding_side='left';tokenizer.truncation_side='left'
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    token_load=time.perf_counter()-started
    torch.cuda.synchronize()
    weight_started=time.perf_counter()
    base=AutoModelForCausalLM.from_pretrained(args.base,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
    torch.cuda.synchronize()
    base_load=time.perf_counter()-weight_started
    memory=None;loading={};memory_load=0
    if args.arm=='md':
        memory_started=time.perf_counter()
        memory=AutoModelForCausalLM.from_pretrained(args.memory,torch_dtype=getattr(torch,args.memory_dtype),
            attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
        torch.cuda.synchronize()
        memory_load=time.perf_counter()-memory_started
    elif args.arm!='base':
        p=args.memory
        memory_started=time.perf_counter()
        memory,loading=MistralMLPModel.from_pretrained(p,config=AutoConfig.from_pretrained(p),
            input_dim=base.config.hidden_size,output_dim=base.config.hidden_size,
            torch_dtype=getattr(torch,args.memory_dtype),output_loading_info=True)
        if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
            raise RuntimeError('Checkpoint class/keys mismatch: '+str(loading))
        memory=memory.to(device).eval()
        torch.cuda.synchronize()
        memory_load=time.perf_counter()-memory_started
    if args.arm=='md':
        from precision.md_runtime import MDRuntime
        runtime=MDRuntime(base,memory,.25)
    else:
        runtime=Runtime(base,memory,args.arm)
    bundle=freeze(tokenizer,args.prompts or PROMPTS)
    props=torch.cuda.get_device_properties(0)
    header=dict(schema='memory-timing-profile-v1',arm=args.arm,memory_dtype=args.memory_dtype,
        measurement_role='shared_bf16_base_reference' if args.arm=='base' else ('fp32_memory' if args.memory_dtype=='float32' else 'bf16_memory'),
        started_at=datetime.now(timezone.utc).isoformat(),
        hardware=dict(gpu_name=props.name,gpu_total_bytes=props.total_memory),
        versions={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','safetensors']},
        python=platform.python_version(),cuda_version=torch.version.cuda,
        policy=dict(base_dtype='bfloat16',memory_dtype=args.memory_dtype,attention='sdpa',kv_cache=True,
            logits_to_keep=1,memory_input='Memory Decoder: own causal LM over same tokens with own KV cache' if args.arm=='md' else 'last decoder MLP pre-hook; last token',
            mlp_lambda=.25,res_lambda=.4,md_lambda=.25,warmups=args.warmups,repeats=args.repeats,smoke=args.smoke),
        weights=dict(base=parameter_inventory(base),memory=parameter_inventory(memory) if memory else None),
        cold_process_load=dict(tokenizer_seconds=token_load,base_seconds=base_load,memory_seconds=memory_load,
            model_load_total_seconds=token_load+base_load+memory_load,
            import_and_cuda_setup_seconds=import_and_cuda_setup_seconds,
            elapsed_since_process_entry_seconds=time.perf_counter()-PROCESS_STARTED,
            filesystem_cache='uncontrolled; no drop_caches',
            allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    write_json(args.out/'header.json',header)
    conditions=[]
    for batch in (1,16):conditions.append(('short_qa',batch,'fixed'))
    if args.smoke:conditions=[('short_qa',1,'fixed')]
    raw=args.out/'requests.jsonl'
    summaries=[]
    for workload,batch,mode in conditions:
        spec=bundle['workloads'][workload]
        ids=torch.tensor(spec[mode+'_input_ids'][:batch],dtype=torch.long)
        masks=torch.tensor(spec[mode+'_attention_mask'][:batch],dtype=torch.long)
        # batch=1 retains the same pre-frozen padded length as batch=16.
        texts=spec['texts'][:batch]
        output_cap=4 if args.smoke else 32
        key=f'{workload}.b{batch}.{mode}'
        rows=[]
        for phase,count in [('first_request',1),('warmup',args.warmups),('steady',args.repeats)]:
            for repeat in range(count):
                gc.collect()
                whole_started=time.perf_counter()
                tokenization_seconds=0
                generated=generate(runtime,ids,masks,max_new_tokens=output_cap,
                    fixed_tokens=True,eos_token_ids=bundle['eos_token_ids'])
                decoded=tokenizer.batch_decode(generated['token_ids'],skip_special_tokens=False)
                row=dict(condition=key,workload=workload,batch_size=batch,mode=mode,phase=phase,repeat=repeat,
                    input_tensor_shape=list(ids.shape),nonpadding_input_tokens=masks.sum(-1).tolist(),
                    max_new_tokens=output_cap,source_ids=spec['source_ids'][:batch],
                    tokenization_seconds=tokenization_seconds,text_e2e_seconds=time.perf_counter()-whole_started,
                    stopped_on_eos=generated['stopped_on_eos'],token_ids=generated['token_ids'],decoded=decoded,
                    **{k:v for k,v in generated['metrics'].items() if k!='batch_size'})
                with raw.open('a') as handle:handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                if phase=='steady':rows.append(row)
                print(json.dumps({k:row[k] for k in ['condition','phase','repeat','elapsed_seconds','generated_tokens','peak_allocated_bytes']}),flush=True)
        summaries.append(dict(condition=key,metrics=aggregate(rows)))
        write_json(args.out/'summary.json',dict(status='in_progress',arm=args.arm,conditions=summaries))
    runtime.close()
    write_json(args.out/'summary.json',dict(status='complete',arm=args.arm,memory_dtype=args.memory_dtype,
        finished_at=datetime.now(timezone.utc).isoformat(),conditions=summaries))


if __name__=='__main__':main()
