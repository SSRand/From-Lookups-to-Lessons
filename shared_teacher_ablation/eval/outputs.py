"""Exact small output closure; every row joins the same checkpoint/input/readout."""
import math
from pathlib import Path
from . import protocol as p
from . import numerics

def batches(rr):
 out=[];batch=[];ds=None
 for r in rr:
  if batch and (len(batch)==16 or r['dataset']!=ds):out.append(batch);batch=[]
  batch.append(r);ds=r['dataset']
 if batch:out.append(batch)
 return out

def identity(c,cid,readout,role,weight,barrier_sha=None):
 model=c['models'][cid]
 return dict(config_sha256=c['_sha256'],source_manifest_sha256=c['source_manifest_sha256'],cohort=c['cohort'],cell=cid,seed=model['seed'],objective=model['objective'],readout=readout,role=role,lambda_=weight,checkpoint_sha256=model['files']['model.safetensors']['sha256'],training_completion_sha256=model['training_completion_sha256'],parent_training_complete_sha256=c['parent']['training_complete_sha256'],population_sha256=c['population']['COMPLETE.json']['sha256'],runtime=c['runtime'],barrier_sha256=barrier_sha)
def validate_row(row,src,ident,cal=False):
 for k in ('dataset','stable_id','input_sha256','gold_mask_sha256'):p.exact(row[k],src[k])
 if row['identity_sha256']!=p.digest(ident):raise ValueError('row identity edge')
 score=numerics.metrics(row['prediction'],src['aliases'])
 for k in ('em','f1'):
  if type(row[k])not in (int,float) or abs(row[k]-score[k])>1e-12:raise ValueError('row scoring')
 nll=row['token_nll']
 if cal:
  if nll is not None:raise ValueError('calibration not PPL-selected')
 else:
  if not isinstance(nll,list) or len(nll)!=len(src['gold_ids']) or any(type(v)not in (int,float) or not math.isfinite(v) or v<0 for v in nll):raise ValueError('complete gold loss mask')
 if type(row['lambda_'])not in (int,float) or row['lambda_']!=ident['lambda_']:raise ValueError('lambda edge')
def validate_bundle(directory,c,cid,readout,role,expected_rows,weights,barrier_sha=None):
 directory=Path(directory);done=p.read(directory/'COMPLETE.json')
 if done['status']!='complete' or type(done['rows'])is not int or done['rows']!=len(expected_rows)*len(weights):raise ValueError('complete bundle count')
 if set(done['artifacts'])!={'predictions.jsonl'} or p.sha(directory/'predictions.jsonl')!=done['artifacts']['predictions.jsonl']:raise ValueError('bundle artifact')
 identities=[identity(c,cid,readout,role,x,barrier_sha) for x in weights];p.exact(done['identities'],identities)
 actual=p.rows(directory/'predictions.jsonl');expected=[(r,ident) for ident in identities for r in expected_rows]
 if len(actual)!=len(expected):raise ValueError('row count')
 for r,(src,ident) in zip(actual,expected):validate_row(r,src,ident,cal=role=='calibration')
 return actual
