"""Original QA rows, compact input identity and fixed-shape batching."""
import hashlib,json,unicodedata
from collections import Counter
from pathlib import Path
from resmem.prompting import build_prompt
from resmem.scoring import row_aliases
ORDER=('nq','triviaqa','webq','hotpotqa','popqa')
COUNTS={'nq':3610,'triviaqa':11313,'webq':2032,'hotpotqa':7405,'popqa':14267}
GOLD={'nq':16640,'triviaqa':58245,'webq':8476,'hotpotqa':34786,'popqa':49629}
def question_group(question):
 return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',question).casefold().split()).encode()).hexdigest()
def batches(rows):
 out=[];batch=[];last=None
 for r in rows:
  if batch and (len(batch)==16 or r['dataset']!=last):out.append(batch);batch=[]
  batch.append(r);last=r['dataset']
 if batch:out.append(batch)
 return out
def pad_inputs(rows,width=384,pad_id=2):
 import torch
 if not rows or any(not r['prompt_ids'] or len(r['prompt_ids'])>width for r in rows):raise ValueError('empty or overlength prompt')
 ids=torch.full((len(rows),width),pad_id,dtype=torch.long);mask=torch.zeros_like(ids)
 for i,r in enumerate(rows):n=len(r['prompt_ids']);ids[i,-n:]=torch.tensor(r['prompt_ids']);mask[i,-n:]=1
 return ids,mask
def compact(row,ordinal,tokenizer):
 prompt=build_prompt(str(row['question']).strip(),prompt_protocol='concise-format-v1')
 ids=tokenizer.encode(prompt,add_special_tokens=True);gold=tokenizer.encode(' '+row['answer'].strip(),add_special_tokens=False)
 if not gold or not ids or len(ids)>384 or tokenizer.eos_token_id in gold:raise ValueError('input geometry differs')
 return dict(ordinal=ordinal,stable_id=row['stable_id'],dataset=row['dataset'],question_group=question_group(row['question']),prompt_ids=ids,gold_ids=gold,aliases=list(row_aliases(row)))
def source_rows(test_root):
 rows=[]
 for ds in ORDER:
  with (Path(test_root)/f'{ds}.jsonl').open() as f:rows.extend(json.loads(line) for line in f if line.strip())
 return rows
def encode_rows(test_root,tokenizer):
 return validate([compact(row,i,tokenizer) for i,row in enumerate(source_rows(test_root))])
def validate(rows):
 if len(rows)!=38627 or [r['ordinal'] for r in rows]!=list(range(38627)):raise ValueError('population/order differs')
 if len({r['stable_id'] for r in rows})!=38627 or len({r['question_group'] for r in rows})!=37419:raise ValueError('duplicate IDs/cluster count')
 if dict(Counter(r['dataset'] for r in rows))!=COUNTS:raise ValueError('dataset counts differ')
 if {d:sum(len(r['gold_ids']) for r in rows if r['dataset']==d) for d in COUNTS}!=GOLD:raise ValueError('gold token counts differ')
 for r in rows:
  if not 0<len(r['prompt_ids'])<=384 or not 0<len(r['gold_ids'])<=71:raise ValueError('input mask')
  if any(type(t)!=int or not 0<=t<32768 for t in r['prompt_ids']+r['gold_ids']):raise ValueError('token type/range')
 if len(batches(rows))!=2416:raise ValueError('batch plan differs')
 return rows
def load(path):
 with Path(path).open() as f:return validate([json.loads(line) for line in f if line.strip()])
