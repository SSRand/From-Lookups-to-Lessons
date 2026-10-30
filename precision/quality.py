"""FP32 vs BF16 memory quality on all 38,627 test questions (Appendix C).

Fixed coefficients on every dataset: lambda=.25 (MLP Memory, Memory Decoder)
and gamma=.4 (ResMem). BF16 Base, SDPA, KV cache, TF32 disabled, batch 16
padded to 384 input tokens, EOS-or-12-token greedy decoding, plus teacher-forced
NLL of every reference-answer token. Launch one process per GPU:

    python -m precision.quality --test-root data/qa/test/frozen --base-model mistralai/Mistral-7B-v0.3 \
        --mlp MLP_CKPT --resmem RESMEM_CKPT --memory-decoder MD_CKPT \
        --output runs/precision/quality --shard 0 --num-shards 8
"""
import argparse,gc,json,math,os
from pathlib import Path
ARMS={'base':{'arm':'base','dtype':None,'lambda':0.},'mlp-fp32':{'arm':'mlp','dtype':'float32','lambda':.25},'mlp-bf16':{'arm':'mlp','dtype':'bfloat16','lambda':.25},'res-fp32':{'arm':'res','dtype':'float32','lambda':.4},'res-bf16':{'arm':'res','dtype':'bfloat16','lambda':.4},'md-fp32':{'arm':'md','dtype':'float32','lambda':.25},'md-bf16':{'arm':'md','dtype':'bfloat16','lambda':.25}}
def strict_load(info):
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('strict checkpoint load differs')
def run(a):
 import torch
 from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
 from resmem.memory import MistralMLPModel
 from resmem.scoring import normalized_metrics
 from qa.runtime import extract_first_answer_line
 from precision.runtime import Runtime,generate,parameter_inventory
 from precision.md_runtime import MDRuntime
 from precision import inputs,inference
 torch.set_num_threads(2);torch.manual_seed(20260921);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(a.device)
 tokenizer=AutoTokenizer.from_pretrained(a.base_model)
 pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id;eos_id=tokenizer.eos_token_id
 rows=inputs.encode_rows(a.test_root,tokenizer);plan=[(i,b) for i,b in enumerate(inputs.batches(rows)) if i%a.num_shards==a.shard]
 paths={'mlp':a.mlp,'res':a.resmem,'md':a.memory_decoder}
 arms=[x for x in a.arms if ARMS[x]['arm']=='base' or paths[ARMS[x]['arm']]]
 out=a.output/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=True)
 base=AutoModelForCausalLM.from_pretrained(a.base_model,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
 inventories={'base':parameter_inventory(base)}
 for label in arms:
  spec=ARMS[label];target=out/(label+'.jsonl')
  if target.exists():continue
  memory=None
  if spec['arm']=='md':
   memory=AutoModelForCausalLM.from_pretrained(paths['md'],torch_dtype=getattr(torch,spec['dtype']),attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
   runtime=MDRuntime(base,memory,spec['lambda'])
  elif spec['arm']!='base':
   path=paths[spec['arm']]
   memory,info=MistralMLPModel.from_pretrained(path,config=AutoConfig.from_pretrained(path),input_dim=base.config.hidden_size,output_dim=base.config.hidden_size,torch_dtype=getattr(torch,spec['dtype']),output_loading_info=True)
   strict_load(info);memory=memory.to(device).eval()
   runtime=Runtime(base,memory,spec['arm'],mlp_lambda=.25,res_lambda=.4)
  else:runtime=Runtime(base,None,'base')
  if memory is not None:inventories[label]=parameter_inventory(memory)
  tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
  try:
   with tmp.open('w') as handle:
    for bi,batch in plan:
     ids,attention=inputs.pad_inputs(batch,384,pad_id)
     result=generate(runtime,ids,attention,max_new_tokens=12,fixed_tokens=False,eos_token_ids=[eos_id])
     losses=inference.gold_prefix(runtime,ids,attention,[r['gold_ids'] for r in batch])
     texts=tokenizer.batch_decode(result['token_ids'],skip_special_tokens=True)
     for src,tokens,eos,text,nll in zip(batch,result['token_ids'],result['stopped_on_eos'],texts,losses,strict=True):
      answer=extract_first_answer_line(text)
      row=dict(ordinal=src['ordinal'],stable_id=src['stable_id'],dataset=src['dataset'],arm=label,batch=bi,token_ids=tokens,stopped_on_eos=eos,decoded=text,answer=answer,scores=normalized_metrics(answer,src['aliases']),token_nll=nll,nll_sum=math.fsum(nll),gold_count=len(src['gold_ids']))
      handle.write(json.dumps(row,ensure_ascii=False)+'\n')
   os.replace(tmp,target)
  finally:
   runtime.close()
   if tmp.exists():tmp.unlink()
  del runtime,memory;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(arm=label,shard=a.shard,batches=len(plan))),flush=True)
 (out/'HEADER.json').write_text(json.dumps(dict(shard=a.shard,num_shards=a.num_shards,arms={k:ARMS[k] for k in arms},parameter_inventories=inventories,batches=[i for i,_ in plan]),indent=1)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 ap.add_argument('--test-root',type=Path,required=True);ap.add_argument('--base-model',default='mistralai/Mistral-7B-v0.3')
 ap.add_argument('--mlp');ap.add_argument('--resmem');ap.add_argument('--memory-decoder')
 ap.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
 ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda:0')
 ap.add_argument('--shard',type=int,default=0);ap.add_argument('--num-shards',type=int,default=8)
 run(ap.parse_args())
if __name__=='__main__':main()
