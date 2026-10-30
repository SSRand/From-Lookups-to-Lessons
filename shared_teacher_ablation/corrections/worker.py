"""One Base-only worker, exact pre-existing diagnostic batches, no subprocesses."""
import argparse,time
from pathlib import Path
from . import core as c
p=c.p
def run(a):
 source=c.source(a.expected_source_manifest_sha256)
 cfg=p.config(a.config,a.expected_config_sha256);root=p.physical(a.output)
 if p.sha(root/'TEST_AUTH.json')!=a.expected_test_auth_sha256:raise ValueError('parent authenticated test proof')
 auth=p.read(root/'TEST_AUTH.json')
 if a.rank not in range(auth['shards']):raise ValueError('rank outside batch plan')
 for k,w in dict(source_manifest_sha256=source,config_sha256=cfg['_sha256'],cohort=cfg['cohort']).items():p.exact(auth[k],w)
 for r in auth['original_records'].values():
  p.exact(p.stat_record(r['path']),r['stat'])
  if r['stat']['size']<=64*1024 and p.sha(r['path'])!=r['sha256']:raise ValueError('small original proof changed')
 rr,_=c.population.load_bundle(Path(cfg['population']['COMPLETE.json']['path']).parent,cfg['population']['COMPLETE.json']['sha256']);rr=rr['diagnostic'];plan=c.batch_plan(rr,auth['shards'])
 p.exact(p.read(root/'BATCH_PLAN.json'),plan);p.exact(p.sha(root/'BATCH_PLAN.json'),auth['batch_plan_sha256'])
 c.barrier.open_seal(a.barrier,a.expected_barrier_sha256,cfg);p.exact(a.expected_barrier_sha256,auth['barrier_sha256'])
 import torch
 from transformers import AutoTokenizer,AutoModelForCausalLM
 torch.set_num_threads(4);torch.manual_seed(20260922);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(cfg['base'],local_files_only=True)
 if tok.pad_token_id is None:tok.pad_token=tok.eos_token
 base,info=AutoModelForCausalLM.from_pretrained(cfg['base'],torch_dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True,output_loading_info=True,low_cpu_mem_usage=True)
 if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError('strict registered Base load')
 base=base.cuda().eval();base.requires_grad_(False)
 if {x.dtype for x in base.parameters()}!={torch.bfloat16}:raise ValueError('Base precision')
 capture={};handle=base.model.layers[-1].mlp.register_forward_pre_hook(lambda m,args,kwargs:capture.update(hidden_states=args[0].detach()),with_kwargs=True)
 dest=root/'base'/str(a.rank);dest.mkdir(parents=True,exist_ok=False);ident=c.base_identity(cfg,a.expected_barrier_sha256,auth['original_complete_sha256'],a.rank,source);count=0;batchcount=0
 try:
  with (dest/'predictions.jsonl').open('xb') as f:
   for batch,g in zip(c.outputs.batches(rr),plan,strict=True):
    if g['rank']!=a.rank:continue
    predictions=c.numerics.generate(base,None,tok,capture,batch,'residual',0.)
    losses=c.numerics.gold_losses(base,None,tok,capture,batch,'residual',0.)
    for src,pred,nll in zip(batch,predictions,losses,strict=True):
     row=dict(dataset=src['dataset'],stable_id=src['stable_id'],input_sha256=src['input_sha256'],gold_mask_sha256=src['gold_mask_sha256'],identity_sha256=p.digest(ident),lambda_=0.,prediction=pred,token_nll=nll,global_batch_id=g['global_batch_id'],geometry_sha256=g['geometry_sha256'],**c.numerics.metrics(pred,src['aliases']))
     c.outputs.validate_row(row,src,ident,False);f.write(p.encoded(row)+b'\n');count+=1
    f.flush();batchcount+=1;p.atomic(dest/'STATUS.json',dict(phase='base_diagnostic',completed_evaluations=count,completed_batches=batchcount,complete=False,epoch=time.time()))
  p.verify_base(cfg)
  p.fresh(dest/'COMPLETE.json',dict(schema='tec-diagnostic-base-shard-v1',status='complete',complete=True,rank=a.rank,identity=ident,rows=count,batches=batchcount,artifacts={'predictions.jsonl':p.sha(dest/'predictions.jsonl')}))
  c.shard(root,cfg,rr,plan,a.rank,a.expected_barrier_sha256,auth['original_complete_sha256'],source)
 except BaseException as e:p.fresh(dest/'FAILURE.json',dict(error=repr(e),complete=False,epoch=time.time()));raise
 finally:handle.remove()
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 for n in ('config','expected-config-sha256','barrier','expected-barrier-sha256','expected-source-manifest-sha256','expected-test-auth-sha256','output'):ap.add_argument('--'+n,required=True)
 ap.add_argument('--rank',type=int,required=True);run(ap.parse_args())
