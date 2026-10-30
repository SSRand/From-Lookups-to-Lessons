"""Frozen native fusion/generation and answer-loss composition, no new model objective."""
import math
from resmem.scoring import normalized_metrics
from resmem.prompting import build_prompt

def fusion(base,memory,method,lambda_,residual_scale=1.0):
 import torch
 if method not in ('mlpmemory','residual') or type(lambda_) not in (int,float) or not 0<=lambda_<=1 or residual_scale!=1.:raise ValueError('readout contract')
 if lambda_==0:return base.float()
 if method=='residual':return base.float()+lambda_*memory.float()
 if lambda_==1:return memory.float()
 return torch.logaddexp(base.float().log_softmax(-1)+math.log1p(-lambda_),memory.float().log_softmax(-1)+math.log(lambda_))
def metrics(text,aliases):
 m=normalized_metrics(text,aliases);return dict(em=m['exact_match'],f1=m['token_f1'])
def encode(question,answer,tokenizer):
 from resmem.answer_nll import encode_prompt_answer
 return encode_prompt_answer(tokenizer,prompt=build_prompt(question,prompt_protocol='concise-format-v1'),answer=answer,max_length=384)
def gold_losses(base,memory,tokenizer,capture,rr,readout,weight):
 import torch
 from resmem.answer_nll import collate_encoded
 examples=[encode(r['question'],r['answer'],tokenizer) for r in rr]
 b=collate_encoded(examples,pad_token_id=tokenizer.pad_token_id)
 with torch.inference_mode():
  out=base(input_ids=b.input_ids.cuda(),attention_mask=b.attention_mask.cuda(),use_cache=False)
  z=out.logits[b.batch_indices,b.causal_positions].float()
  if weight!=0:
   hidden=capture['hidden_states'][b.batch_indices,b.causal_positions].float()
   z=fusion(z,memory(inputs_embeds=hidden),readout,weight)
  loss=-z.log_softmax(-1).gather(-1,b.target_ids.cuda()[:,None]).squeeze(-1)
  if not torch.isfinite(loss).all():raise ValueError('nonfinite gold NLL')
  return [loss[a:b].cpu().tolist() for a,b in b.token_offsets]
def generate(base,memory,tokenizer,capture,rr,readout,weight):
 from types import SimpleNamespace
 from . import generation as g
 g.fuse_next_token_logits=fusion
 examples=[SimpleNamespace(prompt=build_prompt(r['question'],prompt_protocol='concise-format-v1')) for r in rr]
 return g._generate_predictions(base_model=base,memory_model=memory,tokenizer=tokenizer,examples=examples,method='base' if weight==0 else readout,lambda_=weight,residual_scale=1.,capture=capture,max_new_tokens=12,max_input_length=384,batch_size=16,device='cuda')
