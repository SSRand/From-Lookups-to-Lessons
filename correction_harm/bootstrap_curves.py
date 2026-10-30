"""Question-cluster uncertainty for the entire fixed lambda grid; no tuning.

    python -m correction_harm.bootstrap_curves --bank runs/fig4/bank --output runs/fig4/analysis
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from correction_harm.contract import LAMBDAS

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    v = json.loads((a.bank/'bank.json').read_text())
    with np.load(a.bank/'scores.npz') as z:
        base=z['base'].astype(bool);em=z['em'].astype(bool);datasets=z['dataset'];questions=z['question_group'];methods=z['methods'].tolist()
    M,L=len(methods),len(LAMBDAS)
    groups,gi=np.unique(questions,return_inverse=True)
    n=len(base);g=len(groups)
    memberships=defaultdict(set)
    for k,d in zip(gi,datasets):memberships[int(k)].add(str(d))
    strata=defaultdict(list)
    for k,ds in memberships.items():strata[tuple(sorted(ds))].append(k)
    outcome=np.column_stack([np.ones(n),base.astype(float),
                             *[(~base & em[m,l]).astype(float) for m in range(M) for l in range(L)],
                             *[(base & ~em[m,l]).astype(float) for m in range(M) for l in range(L)]])
    summed=np.zeros((g,outcome.shape[1]));np.add.at(summed,gi,outcome)
    rng=np.random.default_rng(20260921)
    reps=2000;results=[]
    for i in range(0,reps,50):
        weights=np.zeros((min(50,reps-i),g))
        for ix in strata.values():
            weights[:,ix]=rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=len(weights))
        results.append(weights@summed)
    boot=np.concatenate(results)
    total,bc=boot[:,0],boot[:,1];bw=total-bc
    corr,harm=boot[:,2:2+M*L],boot[:,2+M*L:2+2*M*L]
    vals=dict(correction_contribution=corr/total[:,None],
              harm_contribution=harm/total[:,None],
              net_em=(corr-harm)/total[:,None],
              correction_rate=corr/bw[:,None],harm_rate=harm/bc[:,None])
    pooled=[]
    for m,method in enumerate(methods):
        for l,lam in enumerate(LAMBDAS):
            cell=next(dict(x) for x in v['curves'] if x['dataset']=='pooled' and x['method']==method and x['lambda_']==lam)
            cell['ci95']={k:np.quantile(arr[:,m*L+l],[.025,.975]).tolist() for k,arr in vals.items()}
            pooled.append(cell)
    report=dict(schema='fig4-fixed-grid-question-cluster-bootstrap-v1',methods=methods,pooled=pooled,
                per_dataset=[x for x in v['curves'] if x['dataset']!='pooled'],
                n=n,base_correct=int(base.sum()),base_wrong=int((~base).sum()),clusters=g,
                bootstrap=dict(replicates=reps,seed=20260921,unit='NFKC + casefold + collapsed-whitespace question cluster',
                               strata='dataset-membership pattern',pairing='Same cluster draws for all methods and all lambdas',
                               scope='Pointwise intervals conditional on fixed checkpoints and frozen dev-selected policy; excludes retraining and dev-selection uncertainty'))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'CURVES.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(a.output/'BOOTSTRAP.npz',methods=np.array(methods),**vals)
    print(json.dumps(dict(status='complete',n=n,clusters=g,replicates=reps)))

if __name__=='__main__':main()
