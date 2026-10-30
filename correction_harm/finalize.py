"""Combine the frozen dev selection with the bootstrapped test curves into Fig. 4 data.

    python -m correction_harm.finalize --selection runs/fig4/SELECTION.json \
        --analysis runs/fig4/analysis --output runs/fig4/data
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from correction_harm.contract import LAMBDAS

def read(path):
    return json.loads(Path(path).read_text())

def write(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selection',type=Path,required=True)
    ap.add_argument('--analysis',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    selection=read(a.selection)
    data=read(a.analysis/'CURVES.json');methods=data['methods']
    data['selected']={m:selection['selected'][m] for m in methods}
    data['schema']='fig4-full-answer-correction-harm-v1'
    data['development']=dict(counts=selection['counts'],rule=selection['selection'],
                              selected=data['selected'],scores=selection['scores'],
                              frozen_at_utc=selection['frozen_at_utc'])
    data['protocol']=dict(population='Pooled five-QA full generated answers; full-answer exact match',
                         decoder='Greedy; 384 input tokens / 12 new tokens; BF16 Base; FP32 MLP Memory/ResMem, BF16 Memory Decoder; no gate',
                         grid=list(LAMBDAS),paired_base='Same frozen Base answer for every method and lambda',
                         dev_source='DPR training QA dev questions, disjoint from all five benchmarks by normalized question',
                         transfer='One scalar per method selected on equal-weight NQ/TriviaQA dev EM, F1 tie-break, then smaller lambda; transferred to all five benchmark datasets')
    data['equal_dataset_macro']=[]
    fields=['correction_contribution','harm_contribution','net_em','base_em','final_em']
    for method in methods:
        for lam in LAMBDAS:
            rr=[r for r in data['per_dataset'] if r['method']==method and r['lambda_']==lam]
            assert len(rr)==5
            data['equal_dataset_macro'].append(dict(method=method,lambda_=lam,**{k:float(np.mean([r[k] for r in rr])) for k in fields}))
    data['selected_summary']={m:next(r for r in data['pooled'] if r['method']==m and r['lambda_']==data['selected'][m]) for m in methods}
    column={m:i*len(LAMBDAS)+LAMBDAS.index(data['selected'][m]) for i,m in enumerate(methods)}
    data['paired_differences']={}
    with np.load(a.analysis/'BOOTSTRAP.npz') as boot:
        for m1,m2 in itertools.combinations(methods,2):
            data['paired_differences'][f'{m2}_minus_{m1}']={k:dict(estimate=data['selected_summary'][m2][k]-data['selected_summary'][m1][k],
                ci95=np.quantile(boot[k][:,column[m2]]-boot[k][:,column[m1]],[.025,.975]).tolist())
                for k in ('correction_contribution','harm_contribution','net_em','correction_rate','harm_rate')}
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/'correction_harm.json',data)
    cells=data['pooled']+data['per_dataset']
    columns=[k for k in cells[0] if k!='ci95']+['dev_selected']
    with (a.output/'correction_harm.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in cells:w.writerow({**{k:r[k] for k in columns if k!='dev_selected'},'dev_selected':r['lambda_']==data['selected'][r['method']]})
    print(json.dumps(dict(selected=data['selected'],
                          summary={m:{k:r[k] for k in ('correction','harm','preserved','still_wrong','net_em')} for m,r in data['selected_summary'].items()},
                          paired_net_em={k:v['net_em'] for k,v in data['paired_differences'].items()}),indent=1))

if __name__=='__main__':main()
