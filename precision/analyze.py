"""Offline paired analysis of the precision quality run; no model loading or new inference.

BF16-minus-FP32 differences in EM, F1 and reference-answer NLL with a
10,000-draw question-cluster bootstrap (stratified by dataset membership,
shared across all arms and contrasts).

    python -m precision.analyze --test-root data/qa/test/frozen --run runs/precision/quality \
        --output runs/precision/QUALITY_ANALYSIS.json
"""
import argparse,json,math
from collections import Counter
from pathlib import Path
from resmem.scoring import normalized_metrics,row_aliases
from precision import inputs,paired
from precision.quality import ARMS
COMPARISONS=[('mlp-bf16','mlp-fp32'),('res-bf16','res-fp32'),('md-bf16','md-fp32'),('md-fp32','base'),('md-bf16','base')]
def load(run,arm,n):
 rows=[]
 for shard in sorted(run.glob('shard-*')):
  path=shard/(arm+'.jsonl')
  if path.exists():
   with path.open() as f:rows+=[json.loads(x) for x in f if x.strip()]
 rows.sort(key=lambda r:r['ordinal'])
 if [r['ordinal'] for r in rows]!=list(range(n)):raise ValueError('missing/duplicate population: '+arm)
 return rows
def main():
 ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
 ap.add_argument('--test-root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args()
 source=inputs.source_rows(a.test_root)
 src=[dict(ordinal=i,stable_id=r['stable_id'],dataset=r['dataset'],question_group=inputs.question_group(r['question'])) for i,r in enumerate(source)]
 aliases=[row_aliases(r) for r in source]
 present=[arm for arm in ARMS if any((s/(arm+'.jsonl')).exists() for s in a.run.glob('shard-*'))]
 result={arm:load(a.run,arm,len(src)) for arm in present}
 summaries={};arrays={}
 for arm,raw in result.items():
  data=[]
  for s,r,al in zip(src,raw,aliases,strict=True):
   if r['stable_id']!=s['stable_id']:raise ValueError('row identity differs')
   m=normalized_metrics(r['answer'],al)
   data.append(dict(dataset=s['dataset'],exact_match=m['exact_match'],token_f1=m['token_f1'],nll_sum=math.fsum(r['token_nll']),gold_count=r['gold_count']))
  summaries[arm]=paired.aggregate(data);arrays[arm]=[[d[k] for k in ('exact_match','token_f1','nll_sum','gold_count')] for d in data]
 comparisons=[(x,y) for x,y in COMPARISONS if x in result and y in result]
 ci=paired.bootstrap(src,arrays,comparisons)
 contrasts={};drift={}
 for newer,older in comparisons:
  key=f'{newer}-minus-{older}';contrasts[key]={}
  for ds in list(inputs.COUNTS)+['macro']:
   a1=summaries[newer]['macro'] if ds=='macro' else summaries[newer]['datasets'][ds];a0=summaries[older]['macro'] if ds=='macro' else summaries[older]['datasets'][ds]
   contrasts[key][ds]={k:a1[k]-a0[k] for k in ('exact_match','token_f1','nll')};contrasts[key][ds]['ppl_ratio']=math.exp(contrasts[key][ds]['nll']);contrasts[key][ds]['conditional_ci95']=ci[key][ds]
  d=[paired.drift(x['token_ids'],y['token_ids']) for x,y in zip(result[older],result[newer])]
  drift[key]=dict(rows=len(d),sequence_identical_count=sum(v['sequence_identical'] for v in d),
                  extracted_answer_identical_count=sum(x['answer']==y['answer'] for x,y in zip(result[older],result[newer])),
                  first_difference_histogram=dict(Counter('identical' if v['first_difference_index'] is None else str(v['first_difference_index']) for v in d)))
 out=dict(schema='precision-quality-analysis-v1',fixed_coefficients={k:v['lambda'] for k,v in ARMS.items() if k in result},
          absolute=summaries,contrasts=contrasts,sequence_drift=drift,
          bootstrap=dict(repeats=10000,seed=20260921,unit='question_group',strata='dataset-membership pattern'),
          limitations=['Fixed checkpoints and coefficients; no training-seed CI or equivalence margin.','Intervals are conditional question-cluster bootstrap, membership-stratified and shared across all arms.'])
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=1)+'\n')
 print('| contrast | macro EM diff (pp) | macro F1 diff (pp) | F1 95% CI (pp) | macro NLL diff |')
 for key,r in contrasts.items():
  v=r['macro'];c=v['conditional_ci95']['token_f1']
  print(f"| {key} | {100*v['exact_match']:.4f} | {100*v['token_f1']:.4f} | [{100*c['low']:.4f}, {100*c['high']:.4f}] | {v['nll']:.6f} |")
 print('| arm | macro F1 |')
 for arm,s in summaries.items():print(f"| {arm} | {100*s['macro']['token_f1']:.2f} |")
if __name__=='__main__':main()
