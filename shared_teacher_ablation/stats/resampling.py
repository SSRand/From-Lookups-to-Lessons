"""One F1 contrast; paired runs and membership-stratified question product weights."""
from collections import defaultdict
import hashlib
import numpy as np

SEEDS=(42,123,456,789,2026,2027,2028,2029)
REPEATS=10000
QUESTION_SEED=20260921
RUN_SEED=20260922

def interval(values,repeats):
 values=np.asarray(values,dtype=np.float64)
 if values.shape!=(repeats,):raise ValueError('all requested draws must be retained')
 bad=~np.isfinite(values);count=int(bad.sum())
 return dict(status='unavailable_invalid_denominator' if count else 'available',requested_draws=repeats,failed_draws=count,failed_draw_indices=np.flatnonzero(bad).tolist(),low=None if count else float(np.quantile(values,.025,method='linear')),high=None if count else float(np.quantile(values,.975,method='linear')))

def weighted_run_means(weights,matrices,counts):
 """Matrices are dataset × group × run; preserve row denominators within clusters."""
 den=weights@counts.T;bad=np.any(~np.isfinite(den)|(den<=0),axis=1)
 per=np.zeros((len(weights),matrices.shape[2]),dtype=np.float64)
 for d,mat in enumerate(matrices):
  with np.errstate(divide='ignore',invalid='ignore'):per+=(weights@mat)/den[:,d,None]
 per/=len(matrices);bad|=np.any(~np.isfinite(per),axis=1);per[bad]=np.nan
 return per,bad

def compute(rows,res,mlp,repeats=REPEATS,keep_draws=False,heartbeat=None):
 if type(repeats)is not int or repeats<1:raise ValueError('positive integer draws required')
 if not rows:raise ValueError('empty population')
 res=np.asarray(res,dtype=np.float64);mlp=np.asarray(mlp,dtype=np.float64)
 for x in (res,mlp):
  if x.shape!=(8,len(rows)) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('finite complete eight-run F1 matrix in [0,1] required')
 if any(type(r.get('dataset'))is not str or not r['dataset'] or type(r.get('question_group'))is not str or not r['question_group'] for r in rows):raise ValueError('question/dataset identity')
 groups=sorted({r['question_group'] for r in rows});gi={g:i for i,g in enumerate(groups)};datasets=sorted({r['dataset'] for r in rows})
 membership=defaultdict(set)
 for r in rows:membership[r['question_group']].add(r['dataset'])
 strata=defaultdict(list)
 for g in groups:strata[tuple(sorted(membership[g]))].append(gi[g])
 strata=[np.array(v,dtype=np.int64) for _,v in sorted(strata.items())]
 matrices=[];counts=[];delta=(res-mlp).T
 for ds in datasets:
  ix=np.array([i for i,r in enumerate(rows) if r['dataset']==ds]);gx=np.array([gi[rows[i]['question_group']] for i in ix]);count=np.zeros(len(groups));mat=np.zeros((len(groups),8))
  np.add.at(count,gx,1);np.add.at(mat,gx,delta[ix]);counts.append(count);matrices.append(mat)
 matrices=np.array(matrices);counts=np.array(counts)
 plain,_=weighted_run_means(np.ones((1,len(groups))),matrices,counts);plain=plain[0]
 q_rng=np.random.default_rng(QUESTION_SEED);s_rng=np.random.default_rng(RUN_SEED)
 run_weights=s_rng.multinomial(8,np.full(8,1/8),size=repeats)
 draws={k:np.full(repeats,np.nan) for k in ('question','paired_run','joint')};denominator_failures=0
 question_digest=hashlib.sha256()
 for start in range(0,repeats,32):
  size=min(32,repeats-start);w=np.zeros((size,len(groups)))
  for ix in strata:w[:,ix]=q_rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=size)
  question_digest.update(w.astype('<i8').tobytes())
  per,bad=weighted_run_means(w,matrices,counts);denominator_failures+=int(bad.sum());v=run_weights[start:start+size]/8
  draws['question'][start:start+size]=per.mean(axis=1)
  draws['paired_run'][start:start+size]=v@plain
  draws['joint'][start:start+size]=np.sum(v*per,axis=1)
  if heartbeat:heartbeat(start+size)
 out=dict(point=float(plain.mean()),paired_run_differences=plain.tolist(),paired_run_sample_sd=float(np.std(plain,ddof=1)),intervals={k:interval(v,repeats) for k,v in draws.items()},normalized_denominator_failed_draws=denominator_failures,seed_order=list(SEEDS),question_groups=len(groups),rows=len(rows),datasets=datasets,draws_requested=repeats,question_rng_seed=QUESTION_SEED,paired_run_rng_seed=RUN_SEED,question_draws_sha256=question_digest.hexdigest(),paired_run_draws_sha256=hashlib.sha256(run_weights.astype('<i8').tobytes()).hexdigest(),quantile='linear percentile .025/.975',units='F1 in [0,1]; multiply contrast by100 for percentage points')
 if keep_draws:out['draws']={k:v.tolist() for k,v in draws.items()}
 return out
