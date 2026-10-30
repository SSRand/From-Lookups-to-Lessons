"""Full gold teacher forcing with the exact same static B x 1 memory geometry."""
def gold_prefix(runtime,input_ids,attention_mask,golds):
 import torch
 if len(golds)!=input_ids.shape[0] or any(not g for g in golds):raise ValueError('gold batch differs')
 with torch.inference_mode():
  device=runtime.device;ids=input_ids.to(device);mask=attention_mask.to(device)
  result=[[] for _ in golds];cache=None
  for t in range(max(map(len,golds))):
   if t:
    active=torch.tensor([t<len(g) for g in golds],device=device)
    ids=torch.tensor([g[t-1] if t<len(g) else 0 for g in golds],device=device)[:,None]
    mask=torch.cat((mask,active.long()[:,None]),-1)
   logits,cache=runtime.forward(ids,mask,cache)
   lp=torch.log_softmax(logits.float(),-1)
   if not torch.isfinite(lp).all():raise ValueError('nonfinite full-vocabulary log probabilities')
   targets=torch.tensor([g[t] if t<len(g) else 0 for g in golds],device=device)
   losses=(-lp.gather(1,targets[:,None])[:,0]).tolist()
   for i,g in enumerate(golds):
    if t<len(g):result[i].append(losses[i])
  return result
