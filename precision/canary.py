"""FP32-to-BF16 memory conversion diagnostic, never a downstream quality-equivalence test.

Greedy 32-token traces with the FP32 memory, then the same loaded parameters cast
to BF16: logit drift along the FP32 token prefix and independent BF16 generation.

    python -m precision.canary --arm res --memory RESMEM_CKPT --out runs/precision/canary.res
"""
import torch
import torch.nn.functional as F
from precision.runtime import parameter_inventory

@torch.inference_mode()
def logit_drift(reference,converted):
 if reference.shape!=converted.shape or reference.ndim<2 or not torch.isfinite(reference).all() or not torch.isfinite(converted).all():raise ValueError('finite matching logits required')
 a=reference.float();b=converted.float();d=b-a
 lp=F.log_softmax(a,-1);lq=F.log_softmax(b,-1);kl=(lp.exp()*(lp-lq)).sum(-1)
 return dict(max_abs=float(d.abs().max()),relative_l2=float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a).clamp_min(1e-30)),mean_kl_fp32_to_bf16=float(kl.mean()),max_kl_fp32_to_bf16=float(kl.max()),top1_equal=int((a.argmax(-1)==b.argmax(-1)).sum()),positions=a.numel()//a.shape[-1],quality_equivalence_established=False)

@torch.inference_mode()
def convert(memory):
 before=parameter_inventory(memory)
 if set(before['parameter_bytes_by_dtype'])!={'torch.float32'}:raise ValueError('conversion reference must be actual FP32 memory')
 error=0.
 for name,p in memory.named_parameters():
  if not torch.isfinite(p).all():raise ValueError('nonfinite source tensor: '+name)
  rounded=p.to(torch.bfloat16)
  if not torch.isfinite(rounded).all():raise ValueError('nonfinite converted tensor')
  error=max(error,float((p-rounded.float()).abs().max()));p.data=rounded
 memory.to(torch.bfloat16)
 after=parameter_inventory(memory)
 if after['parameter_count']!=before['parameter_count'] or before['parameter_payload_bytes']!=2*after['parameter_payload_bytes'] or set(after['parameter_bytes_by_dtype'])!={'torch.bfloat16'}:raise ValueError('conversion inventory differs')
 return dict(before=before,after=after,max_abs_parameter_rounding=error,conversion='tensorwise torch.float32.to(torch.bfloat16)',quality_equivalence_established=False)

@torch.inference_mode()
def trace(runtime,ids,mask,steps=32,forced=None):
 device=runtime.device;ids=ids.to(device);mask=mask.to(device);cache=None;logs=[];tokens=[]
 if forced is not None and (len(forced)!=ids.shape[0] or any(len(x)!=steps for x in forced)):raise ValueError('forced common prefix shape differs')
 forced_tensor=None if forced is None else torch.tensor(forced,device=device)
 for step in range(steps):
  logits,cache=runtime.forward(ids,mask,cache)
  if not torch.isfinite(logits).all():raise ValueError('nonfinite fused logits')
  logs.append(logits.cpu());next_token=logits.argmax(-1) if forced_tensor is None else forced_tensor[:,step];tokens.append(next_token.cpu());ids=next_token[:,None];mask=torch.cat([mask,torch.ones_like(ids)],-1)
 del cache
 return torch.stack(logs,dim=1),torch.stack(tokens,dim=1).tolist()

def sequence_drift(reference,converted):
 if len(reference)!=len(converted) or any(len(a)!=len(b) for a,b in zip(reference,converted)):raise ValueError('equal fixed lengths required')
 return dict(sequences=len(reference),equal_sequences=sum(a==b for a,b in zip(reference,converted)),tokens=sum(map(len,reference)),equal_tokens=sum(x==y for a,b in zip(reference,converted) for x,y in zip(a,b)),reference_tokens=reference,converted_tokens=converted,quality_equivalence_established=False)

def main():
 import argparse,json
 from pathlib import Path
 ap=argparse.ArgumentParser(description=__doc__.splitlines()[0]);ap.add_argument('--arm',choices=['mlp','res'],required=True)
 ap.add_argument('--base',default='mistralai/Mistral-7B-v0.3');ap.add_argument('--memory',required=True)
 ap.add_argument('--prompts',type=Path,default=None);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
 from resmem.memory import MistralMLPModel
 from precision.runtime import Runtime,generate
 from precision.timing_inputs import PROMPTS,freeze
 torch.set_num_threads(2);torch.manual_seed(20260920);torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_device(0);device=torch.device('cuda:0')
 tokenizer=AutoTokenizer.from_pretrained(a.base)
 base=AutoModelForCausalLM.from_pretrained(a.base,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True).to(device).eval()
 memory,info=MistralMLPModel.from_pretrained(a.memory,config=AutoConfig.from_pretrained(a.memory),input_dim=base.config.hidden_size,output_dim=base.config.hidden_size,torch_dtype=torch.float32,output_loading_info=True)
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('strict checkpoint load differs')
 memory=memory.to(device).eval();rt=Runtime(base,memory,a.arm);bundle=freeze(tokenizer,a.prompts or PROMPTS);reference={}
 for workload in ('short_qa',):
  spec=bundle['workloads'][workload];ids=torch.tensor(spec['fixed_input_ids']);mask=torch.tensor(spec['fixed_attention_mask']);logits,tokens=trace(rt,ids,mask);reference[workload]=(logits,tokens,ids,mask)
 rt.close();conversion=convert(memory);rt=Runtime(base,memory,a.arm);results=[]
 for workload,(ref,tokens,ids,mask) in reference.items():
  converted,forced=trace(rt,ids,mask,forced=tokens)
  if forced!=tokens:raise ValueError('canary common prefix changed')
  actual=generate(rt,ids,mask,max_new_tokens=32,fixed_tokens=True,eos_token_ids=bundle['eos_token_ids'])
  results.append(dict(workload=workload,common_fp32_prefix=logit_drift(ref,converted),independent_generation=sequence_drift(tokens,actual['token_ids']),fp32_decoded=tokenizer.batch_decode(tokens,skip_special_tokens=False),bf16_decoded=tokenizer.batch_decode(actual['token_ids'],skip_special_tokens=False)))
 rt.close();a.out.mkdir(parents=True,exist_ok=True)
 (a.out/'CANARY.json').write_text(json.dumps(dict(status='finite_conversion_diagnostic_complete',arm=a.arm,quality_equivalence_established=False,conversion=conversion,conditions=results),indent=1)+'\n')

if __name__=='__main__':main()
