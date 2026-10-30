"""Immutable five-test, disjoint DPR-dev, and predeclared balanced diagnostic rows.

  python -m shared_teacher_ablation.eval.population --test-root DATA/frozen --dev-rows DEV/dev.jsonl --tokenizer BASE --output POP
--test-root holds the frozen nq/triviaqa/webq/hotpotqa/popqa JSONL files (qa.data.build_test_sets);
--dev-rows is dev.jsonl from qa.data.build_dpr_dev.
"""
import argparse,hashlib,unicodedata
from collections import Counter
from pathlib import Path
from . import protocol as p
from . import numerics
def normalized(q):return ' '.join(unicodedata.normalize('NFKC',q).casefold().split())
def qgroup(q):return hashlib.sha256(normalized(q).encode()).hexdigest()
def compact(row,tokenizer):
 from resmem.scoring import row_aliases
 aliases=row_aliases(row)
 answer=str(row.get('answer',aliases[0])).strip()
 prompt=numerics.build_prompt(row['question'],prompt_protocol='concise-format-v1');ids=tokenizer.encode(prompt,add_special_tokens=True)
 if not 0<len(ids)<=384:raise ValueError('prompt longer than native cap or empty')
 gold=numerics.encode(row['question'],answer,tokenizer)
 if tokenizer.eos_token_id in gold.answer_ids:raise ValueError('unexpected goldEOS')
 out=dict(dataset=row['dataset'],stable_id=row['stable_id'],question=row['question'],answer=answer,aliases=aliases,question_group=qgroup(row['question']),source_row_sha256=p.digest(row),prompt_ids=ids,prompt_token_sha256=p.digest(ids),gold_ids=list(gold.answer_ids),gold_input_ids=list(gold.input_ids),gold_causal_positions=list(gold.answer_causal_positions),gold_mask_sha256=p.digest(dict(targets=gold.answer_target_ids,positions=gold.answer_causal_positions,input_ids=gold.input_ids)))
 out['input_sha256']=p.digest(out);return out
def select(rr,counts,label):
 chosen=[]
 for ds,n in counts.items():
  pool=[r for r in rr if r['dataset']==ds]
  if len(pool)<n or len({r['stable_id'] for r in pool})!=len(pool):raise ValueError('population count/duplicate')
  selected=sorted(pool,key=lambda r:(hashlib.sha256(('user-task-20260922|'+label+'|'+ds+'|'+r['stable_id']).encode()).hexdigest(),r['stable_id']))[:n]
  chosen.extend(sorted(selected,key=lambda r:r['stable_id']))
 return chosen
def assert_disjoint(dev,test):
 qs={normalized(r['question']) for r in test};packed={tuple(r['prompt_ids']) for r in test}
 if any(normalized(r['question']) in qs or tuple(r['prompt_ids']) in packed for r in dev):raise ValueError('dev/test question or packed input overlap')
def validate(rr,counts):
 p.exact(dict(Counter(r['dataset'] for r in rr)),counts)
 if len({r['stable_id'] for r in rr})!=len(rr):raise ValueError('duplicate population ID')
 for r in rr:
  if p.digest({k:v for k,v in r.items() if k!='input_sha256'})!=r['input_sha256']:raise ValueError('input digest')
  if r['question_group']!=qgroup(r['question']) or not r['gold_ids']:raise ValueError('question/gold identity')
  if len(r['gold_ids'])!=len(r['gold_causal_positions']) or [r['gold_input_ids'][j+1] for j in r['gold_causal_positions']]!=r['gold_ids']:raise ValueError('gold causal alignment')
 return rr
def load_bundle(root,expected=None):
 root=Path(root)
 if expected is not None and p.sha(root/'COMPLETE.json')!=expected:raise ValueError('population receipt')
 done=p.read(root/'COMPLETE.json');p.exact(done['complete'],True)
 if done['source_manifest_sha256']!=p.source():raise ValueError('population builder source')
 out={}
 for name,counts in [('test',p.COUNTS),('dev',p.DEV),('diagnostic',p.SUBSET)]:
  path=root/(name+'.jsonl')
  if p.sha(path)!=done['artifacts'][path.name]:raise ValueError('population artifact')
  out[name]=validate(p.rows(path),counts)
 assert_disjoint(out['dev'],out['test'])
 test={r['stable_id']:r for r in out['test']}
 if any(test.get(r['stable_id'])!=r for r in out['diagnostic']):raise ValueError('diagnostic is not exact test subset')
 if select(out['test'],p.SUBSET,'diagnostic')!=out['diagnostic']:raise ValueError('diagnostic selection changed')
 return out,done
def prepare(a):
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.tokenizer)
 sources=[Path(a.test_root)/(ds+'.jsonl') for ds in p.COUNTS]
 test=[compact(r,tok) for path in sources for r in p.rows(path)];validate(test,p.COUNTS)
 if len({r['question_group'] for r in test})!=37419 or sum(len(r['gold_ids']) for r in test)!=167776:raise ValueError('actual five-QA clusters/gold differ')
 devraw=p.rows(a.dev_rows)
 p.exact(dict(Counter(r['dataset'] for r in devraw)),{'nq':2050,'triviaqa':2041})
 if any(r.get('split')!='dev' for r in devraw):raise ValueError('original DPR dev only')
 all_dev=[compact(r,tok) for r in devraw];qs={normalized(r['question']) for r in test};packed={tuple(r['prompt_ids']) for r in test}
 eligible=[r for r in all_dev if normalized(r['question']) not in qs and tuple(r['prompt_ids']) not in packed]
 dev=select(eligible,p.DEV,'dev');assert_disjoint(dev,test);validate(dev,p.DEV)
 diag=select(test,p.SUBSET,'diagnostic');root=p.physical(a.output);root.mkdir(parents=True,exist_ok=False)
 for name,rr in [('test',test),('dev',dev),('diagnostic',diag)]:p.write_rows(root/(name+'.jsonl'),rr)
 p.fresh(root/'COMPLETE.json',dict(schema='tec-qa-population-v1',complete=True,source_manifest_sha256=p.source(),test_source_sha256={path.name:p.sha(path) for path in sources},dev_source_sha256=p.sha(a.dev_rows),counts=dict(test=p.COUNTS,dev=p.DEV,diagnostic=p.SUBSET),dev_excluded=len(all_dev)-len(eligible),test_clusters=37419,test_gold_tokens=167776,artifacts={n+'.jsonl':p.sha(root/(n+'.jsonl')) for n in ('test','dev','diagnostic')},selection='sha256 salt|role|dataset|stable_id, then stable_id order'))
 load_bundle(root,p.sha(root/'COMPLETE.json'))
def main():
 a=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 for n in ('test-root','dev-rows','output'):a.add_argument('--'+n,required=True)
 a.add_argument('--tokenizer',default='mistralai/Mistral-7B-v0.3')
 prepare(a.parse_args())
if __name__=='__main__':main()
