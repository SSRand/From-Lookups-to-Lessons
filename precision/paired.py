"""Descriptive paired contrasts; shared question-cluster resampling, fixed lambda."""
import math
from collections import defaultdict
def aggregate(rows):
 def one(rs):
  loss=math.fsum(r['nll_sum'] for r in rs);tokens=sum(r['gold_count'] for r in rs);nll=loss/tokens
  return dict(rows=len(rs),gold_tokens=tokens,nll_sum=loss,nll=nll,ppl=math.exp(nll),exact_match=math.fsum(r['exact_match'] for r in rs)/len(rs),token_f1=math.fsum(r['token_f1'] for r in rs)/len(rs))
 datasets={d:one([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})}
 macro={k:math.fsum(v[k] for v in datasets.values())/len(datasets) for k in ('exact_match','token_f1','nll')};macro['ppl']=math.exp(macro['nll'])
 return dict(datasets=datasets,macro=macro,pooled=one(rows))
def drift(a,b):
 longest=max(len(a),len(b));prefix=0
 for x,y in zip(a,b):
  if x!=y:break
  prefix+=1
 same=a==b
 return dict(sequence_identical=same,common_prefix_length=prefix,first_difference_index=None if same else prefix,agreement_numerator=sum(x==y for x,y in zip(a,b)),agreement_denominator=longest,length_delta=len(b)-len(a))
def bootstrap(rows,arrays,comparisons,repeats=10000,seed=20260921,heartbeat=None):
 import numpy as np
 groups=sorted({r['question_group'] for r in rows});gi={g:i for i,g in enumerate(groups)};datasets=sorted({r['dataset'] for r in rows});membership=defaultdict(set)
 for r in rows:membership[r['question_group']].add(r['dataset'])
 strata=defaultdict(list)
 for g in groups:strata[tuple(sorted(membership[g]))].append(gi[g])
 strata=[np.array(v,dtype=np.int64) for _,v in sorted(strata.items())]
 # Each group may contain multiple rows, including repetitions within a dataset.
 matrices={};counts={}
 for ds in datasets:
  indices=np.array([i for i,r in enumerate(rows) if r['dataset']==ds],dtype=np.int64);gidx=np.array([gi[rows[i]['question_group']] for i in indices]);count=np.zeros(len(groups));np.add.at(count,gidx,1);counts[ds]=count
  for label,values in arrays.items():
   values=np.asarray(values,dtype=np.float64)
   if values.shape!=(len(rows),4) or not np.isfinite(values).all() or np.any(values[:,3]<=0):raise ValueError('bootstrap EM/F1/NLLsum/gold shape')
   mat=np.zeros((len(groups),4));np.add.at(mat,gidx,values[indices]);matrices[(ds,label)]=mat
 output={f'{a}-minus-{b}':{d:[] for d in datasets+['macro']} for a,b in comparisons};rng=np.random.default_rng(seed)
 # Chunking bounds memory; all views and contrasts consume identical weights.
 for start in range(0,repeats,32):
  size=min(32,repeats-start);w=np.zeros((size,len(groups)))
  for ix in strata:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
  summaries={}
  for ds in datasets:
   den=w@counts[ds]
   if np.any(den<=0):raise ValueError('empty sampled dataset')
   for label in arrays:
    raw=w@matrices[(ds,label)];summaries[(ds,label)]=np.column_stack((raw[:,0]/den,raw[:,1]/den,raw[:,2]/raw[:,3]))
  for a,b in comparisons:
   key=f'{a}-minus-{b}';delta=[]
   for ds in datasets:
    x=summaries[(ds,a)]-summaries[(ds,b)];output[key][ds].extend(x.tolist());delta.append(x)
   output[key]['macro'].extend(np.mean(delta,axis=0).tolist())
  if heartbeat:heartbeat(start+size)
 return {name:{ds:{metric:{'low':float(np.quantile(np.array(samples)[:,i],.025)),'high':float(np.quantile(np.array(samples)[:,i],.975))} for i,metric in enumerate(('exact_match','token_f1','nll'))} for ds,samples in byds.items()} for name,byds in output.items()}
