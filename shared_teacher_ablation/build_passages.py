"""CPU-only passage selection and BM25 teacher prompts for the shared-teacher ablation.

Reads the first 65,536 passages of the BM25 corpus (JSONL with id/title/text), ranks
them by a fixed salted SHA-256 of the passage id and keeps the first 32,768 eligible training passages,
then 2,048 validation passages whose titles never appear in training.
Student: [BOS]+64 prefix+32 target tokens. Teacher: same prefix/targets after
'Context:\\n' + first 512 tokens of 5 different-title BM25 passages + '\\n\\nText:\\n'.
"""
import argparse,hashlib,importlib.metadata,json,math,operator,time
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for raw in iter(lambda:f.read(8*1024*1024),b''):h.update(raw)
 return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def read_rows(path,limit=None):
 rows=[]
 with Path(path).open(encoding='utf-8') as f:
  for line in f:
   if limit is not None and len(rows)==limit:break
   if line.strip():rows.append(json.loads(line))
 return rows
def row_bytes(row):return (json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
def fresh(path,value):
 with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def validate_record(row):
 if len(row['target_ids'])!=32 or len(row['student_ids'])!=97 or row['student_ids'][0]!=1 or row['teacher_ids'][0]!=1:raise ValueError('fixed BOS/prefix64/target32 required')
 for kind in ('student','teacher'):
  ids=row[kind+'_ids'];positions=row[kind+'_positions']
  if positions!=list(range(len(ids)-33,len(ids)-1)) or any(type(x) is not int or x<0 for x in ids+positions):raise ValueError('exact causal predictor geometry required')
  if [ids[p+1] for p in positions]!=row['target_ids']:raise ValueError('student/teacher gold targets differ')
 if row['teacher_ids'][-96:]!=row['student_ids'][-96:]:raise ValueError('teacher must retain identical prefix and target')
 docs=row['retrieved_documents']
 if len(docs)!=5 or len({d['id'] for d in docs})!=5 or any(d['title']==row['source_title'] for d in docs):raise ValueError('five distinct other-title retrieval passages required')
 if any(type(d['score']) is not float or not math.isfinite(d['score']) for d in docs):raise ValueError('finite retrieval scores required')

def make_record(doc,corpus,tok,retrieve):
 tokens=tok.encode(doc['text'],add_special_tokens=False)
 if len(tokens)<96:return None
 prefix=tokens[:64];target=tokens[64:96]
 hits,scores=retrieve(tok.decode(prefix))
 if len(hits)!=len(scores):raise ValueError('retrieval indices/scores differ')
 documents=[];seen=set()
 for hit,score in zip(hits,scores):
  hit=operator.index(hit)
  if hit<0 or hit>=len(corpus) or hit in seen:raise ValueError('invalid or repeated retrieval index')
  seen.add(hit);candidate=corpus[hit]
  if candidate['title']==doc['title']:continue
  documents.append(dict(id=candidate['id'],title=candidate['title'],text=candidate['text'],score=float(score)))
  if len(documents)==5:break
 if len(documents)!=5:return None
 evidence=tok.encode('\n\n'.join(d['text'] for d in documents),add_special_tokens=False)[:512]
 teacher_prefix=tok.encode('Context:\n',add_special_tokens=False)+evidence+tok.encode('\n\nText:\n',add_special_tokens=False)+prefix
 student=[tok.bos_token_id]+prefix+target;teacher=[tok.bos_token_id]+teacher_prefix+target
 row=dict(stable_id='wiki:'+doc['id'],source_title=doc['title'],source_sha256=hashlib.sha256(doc['text'].encode()).hexdigest(),student_ids=student,teacher_ids=teacher,student_positions=list(range(len(prefix),len(prefix)+32)),teacher_positions=list(range(len(teacher_prefix),len(teacher_prefix)+32)),target_ids=target,retrieved_documents=documents)
 validate_record(row);return row

def select_records(corpus,tok,retrieve,training_n=32768,validation_n=2048,progress=lambda x:None,old_prefix=None):
 if len({d['id'] for d in corpus})!=len(corpus):raise ValueError('duplicate corpus IDs')
 if any(not isinstance(d['id'],str) or not isinstance(d['title'],str) or not isinstance(d['text'],str) for d in corpus):raise ValueError('original corpus string schema required')
 train=[];valid=[];train_titles=set();rejected={'short_or_retrieval':0,'validation_train_title':0};last=0.
 candidates=sorted(corpus,key=lambda d:hashlib.sha256(f"42:p0c:{d['id']}".encode()).hexdigest())
 for ordinal,doc in enumerate(candidates):
  if len(train)==training_n and doc['title'] in train_titles:rejected['validation_train_title']+=1;continue
  row=make_record(doc,corpus,tok,retrieve)
  if row is None:rejected['short_or_retrieval']+=1;continue
  if len(train)<training_n:
   train.append(row);train_titles.add(row['source_title'])
   if old_prefix is not None and len(train)==len(old_prefix):check_old_prefix(train,old_prefix)
  else:valid.append(row)
  if time.monotonic()-last>30:progress(dict(phase='selecting',ranked_candidates_considered=ordinal+1,training_rows=len(train),validation_rows=len(valid),rejected=rejected));last=time.monotonic()
  if len(train)==training_n and len(valid)==validation_n:break
 if len(train)!=training_n or len(valid)!=validation_n:raise ValueError(f'Insufficient eligible distinct-title data: training={len(train)}/{training_n}, validation={len(valid)}/{validation_n}; no silent reduction')
 if set(x['stable_id'] for x in train)&set(x['stable_id'] for x in valid) or train_titles&{x['source_title'] for x in valid}:raise ValueError('supervised split overlap')
 return dict(training=train,validation=valid,rejected=rejected,ranked_candidates_considered=ordinal+1)

def check_old_prefix(training,old):
 if len(training)<len(old) or any(digest(a)!=digest(b) for a,b in zip(training,old)):raise ValueError('Reference semantic record prefix changed')
 return hashlib.sha256(b''.join(row_bytes(r) for r in training[:len(old)])).hexdigest()

def run(a):
 output=Path(a.output).absolute()
 if output.exists():raise ValueError('fresh output directory required')
 corpus=read_rows(a.corpus,a.corpus_passages);old=read_rows(a.reference_training) if a.reference_training else None
 if len(corpus)!=a.corpus_passages or len({digest(d['text']) for d in corpus})!=a.corpus_passages:raise ValueError('exact unique corpus count required')
 for row in old or []:validate_record(row)
 from transformers import AutoTokenizer
 import bm25s
 tok=AutoTokenizer.from_pretrained(a.tokenizer)
 if tok.bos_token_id!=1:raise ValueError('original Mistral BOS required')
 packages={k:importlib.metadata.version(k) for k in ('bm25s','numpy','scipy','tokenizers','transformers')}
 output.mkdir(parents=True);start=time.monotonic()
 def progress(value):
  v=dict(value,epoch=time.time(),elapsed_seconds=time.monotonic()-start,complete=value.get('phase')=='data_complete')
  tmp=output/'STATUS.json.partial';tmp.write_text(json.dumps(v,sort_keys=True));tmp.replace(output/'STATUS.json');print(json.dumps(v),flush=True)
 try:
  progress(dict(phase='indexing'));index=bm25s.BM25();index.index(bm25s.tokenize([d['text'] for d in corpus],stopwords='en',show_progress=False),show_progress=False)
  def retrieve(query):
   hits,scores=index.retrieve(bm25s.tokenize([query],stopwords='en',show_progress=False),k=64,show_progress=False)
   return hits[0],scores[0]
  selected=select_records(corpus,tok,retrieve,a.training_rows,a.validation_rows,progress,old)
  prefix_sha=check_old_prefix(selected['training'],old) if old else None
  row_index={};artifacts={}
  for split in ('training','validation'):
   name=split+'.jsonl';entries=[]
   with (output/name).open('xb') as f:
    for i,row in enumerate(selected[split]):
     raw=row_bytes(row);f.write(raw);entries.append(dict(row_ordinal=i,stable_id=row['stable_id'],source_title=row['source_title'],row_sha256=digest(row),byte_sha256=hashlib.sha256(raw).hexdigest(),target_offset=i*32,target_count=32))
   artifacts[name]=sha(output/name);row_index[split]=entries
  fresh(output/'ROW_INDEX.json',row_index);artifacts['ROW_INDEX.json']=sha(output/'ROW_INDEX.json')
  provenance=dict(schema='same-rag-expanded-cpu-data-provenance-v1',corpus_sha256=sha(a.corpus),corpus_rows=a.corpus_passages,reference_training_sha256=sha(a.reference_training) if old else None,reference_prefix_byte_sha256=prefix_sha,tokenizer=a.tokenizer,packages=packages,selection='sha256 of salted passage id, eligible prefix',validation_selection='next eligible records excluding all training source titles',retrieval_corpus_scope='all corpus passages; validation targets are supervised-title-disjoint but not retrieval-corpus-unseen',teacher='same BF16 Base with BM25 top64, first5 different-title passages, evidence512; student prefix64/target32',rejected=selected['rejected'],ranked_candidates_considered=selected['ranked_candidates_considered'],eos_appended=False,qa_labels_used=False)
  fresh(output/'PROVENANCE.json',provenance);artifacts['PROVENANCE.json']=sha(output/'PROVENANCE.json')
  done=dict(schema='same-rag-expanded-cpu-data-completion-v1',status='complete',complete=True,epoch=time.time(),training_rows=a.training_rows,validation_rows=a.validation_rows,training_positions=a.training_rows*32,validation_positions=a.validation_rows*32,positions_per_row=32,training_titles=len({r['source_title'] for r in selected['training']}),validation_titles=len({r['source_title'] for r in selected['validation']}),supervised_title_overlap=0,reference_prefix_rows=len(old) if old else 0,reference_prefix_sha256=prefix_sha,artifacts=artifacts,elapsed_seconds=time.monotonic()-start,teacher_cache_built=False)
  fresh(output/'COMPLETE.json',done);progress(dict(phase='data_complete'));return done
 except BaseException as error:
  fresh(output/'FAILURE.json',dict(status='failed',complete=False,epoch=time.time(),error=repr(error)));raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--corpus',required=True,help='BM25 corpus JSONL (id, title, text); the first --corpus-passages rows are used')
 p.add_argument('--corpus-passages',type=int,default=65536)
 p.add_argument('--tokenizer',default='mistralai/Mistral-7B-v0.3')
 p.add_argument('--output',required=True)
 p.add_argument('--training-rows',type=int,default=32768);p.add_argument('--validation-rows',type=int,default=2048)
 p.add_argument('--reference-training',help='optional earlier training.jsonl whose rows must equal the selected prefix')
 print(json.dumps(run(p.parse_args())),flush=True)
