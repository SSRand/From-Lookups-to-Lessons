"""Descriptive paired correction ratios and fixed Base-token-probability bins."""
import math
from collections import defaultdict
import numpy as np
EDGES=(.01,.1,.5,.9)
METRICS=('repair_rate','harm_rate','preservation_rate','delta_EM','delta_F1')
def ratio(n,d):return None if d==0 else float(n/d)
def point(base_em,arm_em,arm_f1,base_f1):
 b=np.asarray(base_em);a=np.asarray(arm_em);af=np.asarray(arm_f1);bf=np.asarray(base_f1)
 if any(x.shape!=b.shape for x in (a,af,bf)) or b.ndim!=1 or not len(b) or not np.isin(b,[0,1]).all() or not np.isin(a,[0,1]).all() or not np.isfinite(np.concatenate((af,bf))).all():raise ValueError('paired complete answer metrics')
 repair=float(np.sum((b==0)&(a==1)));harm=float(np.sum((b==1)&(a==0)));preserved=float(np.sum((b==1)&(a==1)));correct=int(b.sum());wrong=len(b)-correct
 return dict(rows=len(b),base_correct=correct,base_wrong=wrong,repair_count=repair,harm_count=harm,preserved_count=preserved,repair_rate=ratio(repair,wrong),harm_rate=ratio(harm,correct),preservation_rate=ratio(preserved,correct),delta_EM=(repair-harm)/len(b),delta_F1=float(np.mean(af-bf)))
def summaries(rows,base,arm):
 out={}
 for ds in sorted({r['dataset'] for r in rows}):
  ix=[i for i,r in enumerate(rows) if r['dataset']==ds]
  out[ds]=point(np.array([base[i]['em'] for i in ix]),np.array([arm[i]['em'] for i in ix]),np.array([arm[i]['f1'] for i in ix]),np.array([base[i]['f1'] for i in ix]))
 macro={k:None if any(v[k] is None for v in out.values()) else math.fsum(v[k] for v in out.values())/len(out) for k in METRICS}
 pooled=point(np.array([x['em'] for x in base]),np.array([x['em'] for x in arm]),np.array([x['f1'] for x in arm]),np.array([x['f1'] for x in base]))
 return dict(datasets=out,macro=macro,pooled=pooled)
def probability_bins(prob):
 prob=np.asarray(prob)
 if not np.isfinite(prob).all() or np.any(prob<0) or np.any(prob>1):raise ValueError('valid token probability')
 return np.searchsorted(np.array(EDGES),prob,side='right')
def token_bins(rows,base,arm):
 if not(len(rows)==len(base)==len(arm)):raise ValueError('gold row pairing')
 out={}
 for ds in sorted({r['dataset'] for r in rows}):
  bi=[];ai=[];rowid=[]
  for i,r in enumerate(rows):
   if r['dataset']!=ds:continue
   b=base[i]['token_nll'];a=arm[i]['token_nll']
   if len(b)!=len(a) or not b:raise ValueError('complete aligned gold tokens')
   bi.extend(b);ai.extend(a);rowid.extend([i]*len(b))
  b=np.asarray(bi);a=np.asarray(ai);idx=np.asarray(rowid)
  if not np.isfinite(b).all() or not np.isfinite(a).all() or np.any(b<0) or np.any(a<0):raise ValueError('finite nonnegative gold losses')
  which=probability_bins(np.exp(-b));bins=[]
  for k in range(5):
   mask=which==k;n=int(mask.sum());delta=a[mask]-b[mask];unique=set(idx[mask].tolist())
   bins.append(dict(bin=k,lower=[0,*EDGES][k],upper=[*EDGES,1][k],upper_inclusive=k==4,gold_tokens=n,question_rows=len(unique),question_clusters=len({rows[i]['question_group'] for i in unique}),base_nll_sum=math.fsum(b[mask]),arm_nll_sum=math.fsum(a[mask]),delta_nll_sum=math.fsum(delta),mean_delta_nll=ratio(math.fsum(delta),n),improved_tokens=int(np.sum(delta<0)),worsened_tokens=int(np.sum(delta>0)),unchanged_tokens=int(np.sum(delta==0)),improved_fraction=ratio(int(np.sum(delta<0)),n),worsened_fraction=ratio(int(np.sum(delta>0)),n)))
  out[ds]=bins
 return out
def bootstrap(rows,base_em,base_f1,cells,repeats=10000,seed=20260921,heartbeat=None):
 # cells contain seed-averaged *indicators*, not thresholded mean EM.
 groups=sorted({r['question_group'] for r in rows});gi={g:i for i,g in enumerate(groups)};ds=sorted({r['dataset'] for r in rows});membership=defaultdict(set)
 for r in rows:membership[r['question_group']].add(r['dataset'])
 strata=defaultdict(list)
 for g in groups:strata[tuple(sorted(membership[g]))].append(gi[g])
 strata=[np.array(v,dtype=np.int64) for _,v in sorted(strata.items())]
 b=np.asarray(base_em,dtype=float);bf=np.asarray(base_f1,dtype=float)
 if b.shape!=(len(rows),) or bf.shape!=b.shape or not np.isin(b,[0,1]).all():raise ValueError('Base paired binary EM')
 den={};num={}
 for d in ds:
  ix=np.array([i for i,r in enumerate(rows) if r['dataset']==d]);gidx=np.array([gi[rows[i]['question_group']] for i in ix]);dm=np.zeros((len(groups),5))
  np.add.at(dm,gidx,np.column_stack((1-b[ix],b[ix],b[ix],np.ones(len(ix)),np.ones(len(ix)))));den[d]=dm
  for label,(ae,af) in cells.items():
   ae=np.asarray(ae);af=np.asarray(af)
   if ae.shape!=b.shape or af.shape!=b.shape or not np.isfinite(ae).all() or not np.isfinite(af).all() or np.any(ae<0) or np.any(ae>1):raise ValueError('finite seed-averaged metrics')
   values=np.column_stack(((1-b)*ae,b*(1-ae),b*ae,ae-b,af-bf));mat=np.zeros((len(groups),5));np.add.at(mat,gidx,values[ix]);num[d,label]=mat
 sample={label:{d:[] for d in ds+['macro']} for label in cells};rng=np.random.default_rng(seed)
 for start in range(0,repeats,32):
  count=min(32,repeats-start);w=np.zeros((count,len(groups)))
  for ix in strata:w[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=count)
  denominators={d:w@den[d] for d in ds}
  for label in cells:
   values=[]
   for d in ds:
    numerator=w@num[d,label];denominator=denominators[d];v=np.full_like(numerator,np.nan);np.divide(numerator,denominator,out=v,where=denominator>0);sample[label][d].append(v);values.append(v)
   sample[label]['macro'].append(np.mean(values,axis=0))
  if heartbeat:heartbeat(start+count)
 out={}
 for label,dd in sample.items():
  out[label]={}
  for d,chunks in dd.items():
   values=np.concatenate(chunks);out[label][d]={}
   for i,k in enumerate(METRICS):
    x=values[:,i];invalid=int(np.sum(~np.isfinite(x)))
    out[label][d][k]=dict(low=None if invalid else float(np.quantile(x,.025)),high=None if invalid else float(np.quantile(x,.975)),invalid_draws=invalid,reason='resampled_empty_denominator' if invalid else None)
 return out
