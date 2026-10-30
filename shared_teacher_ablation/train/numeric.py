"""Original native loss and FP32 AdamW path with explicit fixed budget inputs."""
import copy,math,random
import numpy as np
import torch
from .protocol import learning_rate

def objective(memory,base,ids,probs,gold,kind,alpha=.5):
 if type(alpha) not in (float,int) or not math.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('KL weight alpha must be finite in [0,1]')
 if kind not in ('standalone_memory','joint_base_anchored'):raise ValueError('Unknown objective')
 logits=memory if kind=='standalone_memory' else memory+base
 logp=torch.nn.functional.log_softmax(logits.float(),dim=-1)
 kl=(probs*(probs.clamp_min(1e-30).log()-logp.gather(-1,ids.long()))).sum(-1).mean()
 ce=torch.nn.functional.nll_loss(logp,gold.long())
 return alpha*kl+(1-alpha)*ce,kl,ce

def batch(arrays,indices,device):
 result={}
 for name in arrays:
  x=torch.from_numpy(np.array(arrays[name][indices],copy=True))
  if name in ('hidden','base_logits'):x=x.view(torch.bfloat16).float()
  result[name]=x.to(device)
 return result

def forward(model,values,kind,alpha):
 return objective(model(inputs_embeds=values['hidden']) if hasattr(model,'layers') else model(values['hidden']),values['base_logits'],values['teacher_ids'],values['teacher_probs'],values['gold'],kind,alpha)

def sample_parameters(model):
 out={}
 for name,value in model.named_parameters():
  x=value.detach().reshape(-1);n=min(64,x.numel());ids=torch.arange(n,dtype=torch.int64,device=x.device)*(x.numel()-1)//max(1,n-1)
  out[name]=x.index_select(0,ids).float().cpu().clone()
 return out

def step_update(model,opt,values,kind,step,total,micro=32,alpha=.5,sample=False):
 n=len(values['gold'])
 if n%micro:raise ValueError('Partial microbatch')
 opt.zero_grad(set_to_none=True);rate=learning_rate(step,total);before=sample_parameters(model) if sample else None
 for g in opt.param_groups:g['lr']=rate
 sums=np.zeros(3)
 for j in range(0,n,micro):
  loss,kl,ce=forward(model,{k:v[j:j+micro] for k,v in values.items()},kind,alpha)
  if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
  factor=micro/n;(loss*factor).backward();sums+=np.array([float(x.detach()) for x in (loss,kl,ce)])*factor
 norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
 if not math.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
 opt.step();update=None
 if before is not None:
  after=sample_parameters(model);d={k:float(torch.linalg.vector_norm(after[k]-before[k])) for k in before}
  if not all(math.isfinite(x) for x in d.values()):raise FloatingPointError('Nonfinite parameter sample')
  update=dict(l2=math.sqrt(sum(x*x for x in d.values())),parameters=d,finite=True,scope='64 fixed evenly-spaced entries per tensor; actual FP32 update, not full norm')
 return dict(loss=float(sums[0]),kl=float(sums[1]),ce=float(sums[2]),grad_norm=norm,lr=rate,update_sample=update)

def save_state(opt,step,identity,order_sha,n,batch_size,cuda=True,copy_optimizer=True):
 per_epoch=n//batch_size
 return dict(schema='teacher-exposure-optimizer-rng-v1',optimizer=copy.deepcopy(opt.state_dict()) if copy_optimizer else opt.state_dict(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng_cpu=torch.get_rng_state(),torch_rng_cuda=torch.cuda.get_rng_state() if cuda else None,step=step,next_epoch=step//per_epoch,next_batch_offset=(step%per_epoch)*batch_size,identity=identity,order_sha256=order_sha,n_positions=n,batch=batch_size)

def restore_state(opt,state,identity,order_sha,n,batch_size,total,cuda=True):
 per_epoch=n//batch_size;s=state.get('step')
 if state.get('schema')!='teacher-exposure-optimizer-rng-v1' or state.get('identity')!=identity or state.get('order_sha256')!=order_sha or state.get('n_positions')!=n or state.get('batch')!=batch_size:raise ValueError('Resume identity/order mismatch')
 if type(s) is not int or not 0<=s<total or state.get('next_epoch')!=s//per_epoch or state.get('next_batch_offset')!=(s%per_epoch)*batch_size:raise ValueError('Resume cursor invalid or already terminal')
 opt.load_state_dict(state['optimizer']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng_cpu'])
 for values in opt.state.values():
  for value in values.values():
   if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):raise ValueError('Nonfinite restored optimizer state')
 if cuda:
  if state['torch_rng_cuda'] is None:raise ValueError('Missing CUDA RNG')
  torch.cuda.set_rng_state(state['torch_rng_cuda'])
 return s
