"""Paired-run and crossed (question x run) intervals for the native RES-minus-MLP macro F1 contrast. CPU only.

  CUDA_VISIBLE_DEVICES= python -m shared_teacher_ablation.stats.run --snapshot EVAL/test [--snapshot ...] \\
      --calibration EVAL/calibration [--calibration ...] --population-root POP --barrier SEAL.json \\
      --analysis ANALYSIS/ANALYSIS.json --output SEEDCI
"""
import argparse,datetime,os,platform,sys,time
from pathlib import Path
import numpy as np
from . import evidence as e
from .resampling import compute

LIMITS=[
 'The question-cluster primary interval and decision rule remain unchanged; all supplementary views are reported regardless of sign.',
 'Eight prespecified paired runs are an empirical sample, not a demonstrated random sample from all training seeds. Small-n percentile coverage is approximate.',
 'Joint product-weight bootstrap is crossed, not nested; no exact joint 95% coverage guarantee or equivalence claim.',
 'Pair resampling does not separately identify initialization, data order, selected policy or residual hardware variability.',
 'Fixed teacher, training data, runtime, dev population and five benchmarks; no retraining, recalibration or model re-inference.',
]

def main(a):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit CUDA_VISIBLE_DEVICES empty required')
 out=e.physical(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  rows,res,mlp,analysis,inputs=e.load_inputs(a,out)
  e.write(out/'INPUTS.json',inputs)
  def heartbeat(draws):
   q=out/'STATUS.json';tmp=out/'STATUS.tmp';e.write(tmp,dict(phase='shared_bootstrap',draws=draws,epoch=time.time(),complete=False));os.replace(tmp,q)
  result=compute(rows,res,mlp,heartbeat=heartbeat);match=e.match_primary(result,analysis)
  unavailable=any(x['status']!='available' for x in result['intervals'].values())
  result.update(schema='paired-run-question-uncertainty-v1',status='computed_with_unavailable_intervals' if unavailable else 'computed',scientific_acceptance=False,metric='native_RES_minus_MLP_equal_dataset_macro_F1',primary_unchanged=True,original_primary_reproduction=match,inputs=inputs,limits=LIMITS)
  e.write(out/'RESULTS.json',result)
  labels={'question':'question-cluster (primary)','paired_run':'paired-run (supplementary)','joint':'crossed question x run (supplementary)'}
  report=['# Paired-run uncertainty for the shared-teacher ablation','',f"Status: {result['status']}.",'',f"Reproduced primary point estimate: {result['point']*100:.6f} F1 points.",'','| View | Two-sided 95% interval (points) | Invalid/requested draws |','|---|---|---|']
  for k,v in result['intervals'].items():
   bounds=f"[{100*v['low']:.6f}, {100*v['high']:.6f}]" if v['status']=='available' else 'unavailable; all failed draws retained, none discarded or redrawn'
   report.append(f"| {labels[k]} | {bounds} | {v['failed_draws']}/{v['requested_draws']} |")
  report+=['','| Paired seed | RES - MLP (F1 points) |','|---|---|']
  report += [f'| {s} | {d*100:.6f} |' for s,d in zip(result['seed_order'],result['paired_run_differences'])]
  report += ['',f"Sample SD of the eight paired differences: {result['paired_run_sample_sd']*100:.6f} points (not a CI of the mean).",'','Empirical sensitivity analysis over eight specified runs; the primary interval is retained and is not replaced by a supplementary view, and an interval containing zero does not imply equivalence.','',*['- '+x for x in LIMITS],'']
  with (out/'REPORT.md').open('x') as f:f.write('\n'.join(report))
  artifacts={x:e.digest(e.safe_file(out/x)) for x in ('INPUTS.json','REBUILT_GLOBAL_BARRIER.json','RESULTS.json','REPORT.md')}
  e.write(out/'OFFLINE_RESULT.json',dict(status=result['status'],scientific_acceptance=False,started_utc=start,elapsed_seconds=time.monotonic()-started,python=sys.version,numpy=np.__version__,platform=platform.platform(),thread_environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},artifacts=artifacts))
  q=out/'STATUS.json';tmp=out/'STATUS.tmp';e.write(tmp,dict(phase='complete',complete=True,scientific_acceptance=False,epoch=time.time()));os.replace(tmp,q)
 except BaseException as error:
  e.write(out/'FAILURE.json',dict(status='failed_preserved',scientific_acceptance=False,error=repr(error),elapsed_seconds=time.monotonic()-started));raise

def parser():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--snapshot',action='append',required=True,help='test phase root of each cohort');p.add_argument('--calibration',action='append',required=True,help='calibration phase root of each cohort')
 for name in ('population-root','barrier','analysis','output'):p.add_argument('--'+name,required=True)
 return p
if __name__=='__main__':main(parser().parse_args())
