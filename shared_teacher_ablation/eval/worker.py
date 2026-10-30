"""One exact fresh model/GPU; foreground, no subprocesses or cache loading."""
import argparse,time
from pathlib import Path
from . import protocol as p
from . import population,numerics,outputs,barrier

def run(a):
 c=p.config(a.config,a.expected_config_sha256);m=c['models'][a.cell]
 if a.rank not in range(len(c['models'])) or sorted(c['models'])[a.rank]!=a.cell:raise ValueError('fixed cell/rank')
 root=Path(c['output'])/a.phase;pop,_=population.load_bundle(Path(c['population']['COMPLETE.json']['path']).parent,c['population']['COMPLETE.json']['sha256'])
 seal=barrier.open_seal(a.barrier,a.expected_barrier_sha256,c) if a.phase=='test' else None
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
 from resmem.memory import MistralMLPModel
 torch.set_num_threads(4);torch.manual_seed(m['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(c['base'],local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 base,info=AutoModelForCausalLM.from_pretrained(c['base'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,output_loading_info=True,low_cpu_mem_usage=True)
 strict(info);base=base.cuda().eval();base.requires_grad_(False)
 mem,info=MistralMLPModel.from_pretrained(m['checkpoint'],config=AutoConfig.from_pretrained(m['checkpoint'],local_files_only=True),input_dim=4096,output_dim=4096,torch_dtype=torch.float32,local_files_only=True,output_loading_info=True)
 strict(info);mem=mem.cuda().eval();mem.requires_grad_(False)
 if sum(x.numel() for x in mem.parameters())!=1543540736 or {x.dtype for x in mem.parameters()}!={torch.float32} or {x.dtype for x in base.parameters()}!={torch.bfloat16}:raise ValueError('loaded precision/architecture')
 capture={};handle=base.model.layers[-1].mlp.register_forward_pre_hook(lambda module,args,kwargs:capture.update(hidden_states=args[0].detach()),with_kwargs=True)
 total=0
 def evaluate(readout,role,rr,weights,destination):
  nonlocal total
  destination.mkdir(parents=True,exist_ok=False);identities=[outputs.identity(c,a.cell,readout,role,w,a.expected_barrier_sha256 if seal else None) for w in weights]
  try:
   count=0
   with (destination/'predictions.jsonl').open('xb') as f:
    for ident in identities:
     weight=ident['lambda_']
     for batch in outputs.batches(rr):
      predictions=numerics.generate(base,mem,tok,capture,batch,readout,weight)
      losses=[None]*len(batch) if role=='calibration' else numerics.gold_losses(base,mem,tok,capture,batch,readout,weight)
      for src,pred,nll in zip(batch,predictions,losses,strict=True):
       row=dict(dataset=src['dataset'],stable_id=src['stable_id'],input_sha256=src['input_sha256'],gold_mask_sha256=src['gold_mask_sha256'],identity_sha256=p.digest(ident),lambda_=weight,prediction=pred,token_nll=nll,**numerics.metrics(pred,src['aliases']))
       outputs.validate_row(row,src,ident,role=='calibration');f.write(p.encoded(row)+b'\n');count+=1;total+=1
      f.flush();p.atomic(root/(a.cell+'.STATUS.json'),dict(phase=a.phase,cell=a.cell,readout=readout,role=role,lambda_=weight,completed_evaluations=total,epoch=time.time(),complete=False))
   p.fresh(destination/'COMPLETE.json',dict(status='complete',identities=identities,rows=count,artifacts={'predictions.jsonl':p.sha(destination/'predictions.jsonl')},epoch=time.time()))
   outputs.validate_bundle(destination,c,a.cell,readout,role,rr,weights,a.expected_barrier_sha256 if seal else None)
  except BaseException as e:p.fresh(destination/'FAILURE.json',dict(error=repr(e),epoch=time.time()));raise
 try:
  if a.phase=='calibration':
   for readout in p.READOUTS:evaluate(readout,'calibration',pop['dev'],p.GRID,root/'models'/a.cell/readout)
  else:
   native=p.NATIVE[m['objective']];other=next(r for r in p.READOUTS if r!=native)
   weights={r:seal['policies'][p.key(m['seed'],m['objective'],r)]['selected'] for r in p.READOUTS}
   evaluate(native,'native_selected',pop['test'],[weights[native]],root/'models'/a.cell/'native_selected')
   # All workers share the same full-population batch plan; each cohort assembles one exact Base.
   base_rows=[r for i,b in enumerate(outputs.batches(pop['test'])) if i%len(c['models'])==a.rank for r in b]
   evaluate(native,'base_shard',base_rows,[0.],root/'base'/str(a.rank))
   evaluate(native,'native_selected_diagnostic',pop['diagnostic'],[weights[native]],root/'models'/a.cell/'native_selected_diagnostic')
   evaluate(other,'cross_selected',pop['diagnostic'],[weights[other]],root/'models'/a.cell/'cross_selected')
   for readout in p.READOUTS:evaluate(readout,'fixed1',pop['diagnostic'],[1.],root/'models'/a.cell/('fixed1-'+readout))
  for r in m['files'].values():p.verify(r)
  p.atomic(root/(a.cell+'.STATUS.json'),dict(phase=a.phase+'_complete',completed_evaluations=total,epoch=time.time(),complete=True))
 finally:handle.remove()
def strict(info):
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('strict model load')
def main():
 ap=argparse.ArgumentParser()
 for n in ('config','expected-config-sha256','cell'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--phase',choices=['calibration','test'],required=True);ap.add_argument('--rank',type=int,required=True);ap.add_argument('--barrier');ap.add_argument('--expected-barrier-sha256');run(ap.parse_args())
if __name__=='__main__':main()
