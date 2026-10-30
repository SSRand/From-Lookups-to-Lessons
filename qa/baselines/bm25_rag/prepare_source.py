"""Restore exact training body coverage before building same-source passages.

The BM25-RAG corpus is the Wikipedia text used for memory training: the Atlas
December 2021 dump (text-list-100-sec.jsonl), minus the seeded held-out 0.2%
(numpy default_rng(42) permutation, as in the MLP Memory preprocessing), and
minus the tail tokens dropped when grouping into 1,024-token training rows.
Retained coverage is verified batch by batch against the tokenized training
set (Rubin-Wei/enwiki-dec2021-preprocessed-mistral), then converted into
non-overlapping 100-word passages within each article.

    python -m qa.baselines.bm25_rag.prepare_source tokenize --source text-list-100-sec.jsonl --output CORPUS_DIR
    python -m qa.baselines.bm25_rag.prepare_source arrows --arrow-dir TRAIN_ARROW_DIR --output CORPUS_DIR
    python -m qa.baselines.bm25_rag.prepare_source align --source text-list-100-sec.jsonl --output CORPUS_DIR
    python -m qa.baselines.bm25_rag.prepare_source convert --source text-list-100-sec.jsonl --output CORPUS_DIR
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from qa.baselines.bm25_rag.corpus import align_scope,iter_chunks

SOURCE=ARROWS=OUT=None
MODEL='mistralai/Mistral-7B-v0.3'
N=33176581
SOURCE_URL='https://dl.fbaipublicfiles.com/atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl'
SOURCE_SHA256='a21b3f1374a8a795e4412f436b1555f3cba155c9cdb008ae37ec2093f1d32485'
TRAIN_REPO='Rubin-Wei/enwiki-dec2021-preprocessed-mistral'
TRAIN_REVISION='2655b013f34bfbea6d113b33089caf100cf4cdb0'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dump(p,v):
    p=Path(p);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def raw_worker(job):
    part,start,end,identity,source,model,out=job
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model)
    root=out/'raw_parts'/str(part);root.mkdir(parents=True,exist_ok=True)
    if (root/'DONE.json').exists():
        cached=json.loads((root/'DONE.json').read_text())
        if cached.get('identity')!=identity or cached.get('byte_range')!=[start,end]:
            raise ValueError('Raw cached partition/source/tokenizer identity mismatch')
        return cached
    ids=[];positions=[];lengths=[];offsets=[];count=0;token_count=0
    with source.open('rb') as src,(root/'tokens.u32').open('wb') as dst:
        if start: src.seek(start-1);src.readline()
        else: src.seek(0)
        while src.tell()<end:
            batch=[];batch_offsets=[]
            for _ in range(4096):
                if src.tell()>=end: break
                batch_offsets.append(src.tell());line=src.readline()
                if not line: break
                batch.append(json.loads(line))
            if not batch: break
            encoded=tok([r['text'] for r in batch],add_special_tokens=True,return_attention_mask=False)['input_ids']
            for row,off,tokens in zip(batch,batch_offsets,encoded,strict=True):
                rid=int(row['id'])
                if ids and rid!=ids[-1]+1: raise ValueError('Raw IDs are not consecutive')
                ids.append(rid);positions.append(token_count);lengths.append(len(tokens));offsets.append(off)
                np.asarray(tokens,dtype='<u4').tofile(dst);token_count+=len(tokens);count+=1
    np.save(root/'index.npy',np.array([positions,lengths,offsets],dtype=np.uint64).T)
    receipt=dict(part=part,identity=identity,byte_range=[start,end],first_id=ids[0],last_id=ids[-1],rows=count,tokens=token_count,
                 artifacts={n:sha(root/n) for n in ['tokens.u32','index.npy']})
    dump(root/'DONE.json',receipt);return receipt

def tokenize():
    OUT.mkdir(parents=True,exist_ok=True)
    if SOURCE.stat().st_size!=20900665949: raise ValueError('Wrong raw source byte count')
    size=SOURCE.stat().st_size;workers=32
    identity=dict(source_sha256=sha(SOURCE),tokenizer=MODEL)
    if identity['source_sha256']!=SOURCE_SHA256: raise ValueError('Wrong raw source bytes')
    jobs=[(i,size*i//workers,size*(i+1)//workers,identity,SOURCE,MODEL,OUT) for i in range(workers)]
    started=time.time();receipts=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(raw_worker,jobs):
            receipts.append(r);print(json.dumps(dict(stage='tokenize',part=r['part'],rows=r['rows'],elapsed=time.time()-started)),flush=True)
    if sum(r['rows'] for r in receipts)!=N: raise ValueError('Wrong raw source population')
    index=np.lib.format.open_memmap(OUT/'raw_index.npy',mode='w+',dtype=np.uint64,shape=(N,3))
    tokens=0;rows=0
    with (OUT/'raw_tokens.u32').open('wb') as out:
        import shutil
        for r in receipts:
            if r['first_id']!=rows or r['last_id']!=rows+r['rows']-1: raise ValueError('Raw partition gap')
            root=OUT/'raw_parts'/str(r['part'])
            for n,h in r['artifacts'].items():
                if sha(root/n)!=h: raise ValueError('Raw part changed')
            a=np.load(root/'index.npy');a[:,0]+=tokens;index[rows:rows+r['rows']]=a
            with (root/'tokens.u32').open('rb') as src: shutil.copyfileobj(src,out,8*1024*1024)
            rows+=r['rows'];tokens+=r['tokens']
    index.flush();del index
    dump(OUT/'RAW_VERIFIED.json',dict(rows=rows,tokens=tokens,source_sha256=sha(SOURCE),
        source_bytes=size,source_url=SOURCE_URL,tokenizer=MODEL,
        artifacts={n:sha(OUT/n) for n in ['raw_index.npy','raw_tokens.u32']},parts=receipts))

def arrow_worker(record):
    import pyarrow.ipc as ipc
    path=record['path'];part=int(path.name.split('-')[1])
    root=record['out']/'arrow_parts'/str(part);root.mkdir(parents=True,exist_ok=True)
    record['sha256']=sha(path)
    if (root/'DONE.json').exists():
        cached=json.loads((root/'DONE.json').read_text())
        if cached.get('source_sha256')!=record['sha256'] or cached.get('part')!=part:
            raise ValueError('Arrow cached source identity mismatch')
        return cached
    t=ipc.open_stream(path).read_all()
    ids=t['input_ids'].combine_chunks().values.to_numpy().reshape(-1,2048)
    labels=t['labels'].combine_chunks().values.to_numpy().reshape(-1,2048)
    mask=labels!=-100
    if not np.array_equal(ids[mask],labels[mask]): raise ValueError('Labels and input differ')
    lengths=mask.sum(axis=1)
    if np.any(lengths<=0) or np.any(lengths>1024): raise ValueError('Unexpected retained row length')
    reset=mask[:,0]
    for i in range(len(ids)):
        expected=np.zeros(2048,dtype=bool)
        if reset[i]: expected[:lengths[i]]=True
        else: expected[1024:1024+lengths[i]]=True
        if not np.array_equal(mask[i],expected): raise ValueError('Unexpected overlap/padding layout')
    tokens=ids[mask].astype('<u4');tokens.tofile(root/'tokens.u32')
    prefix=np.r_[0,np.cumsum(lengths)]
    starts=prefix[:-1][reset]
    np.save(root/'resets.npy',starts.astype(np.uint64))
    receipt=dict(part=part,rows=len(ids),tokens=len(tokens),source_sha256=record['sha256'],
                 artifacts={n:sha(root/n) for n in ['tokens.u32','resets.npy']})
    dump(root/'DONE.json',receipt);return receipt

def arrows():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[dict(name=p.name,path=p,out=OUT) for p in sorted(ARROWS.glob('data-*.arrow'),key=lambda p:p.name)];receipts=[]
    with ProcessPoolExecutor(max_workers=16) as pool:
        for r in pool.map(arrow_worker,records):
            receipts.append(r)
            if len(receipts)%100==0: print(json.dumps(dict(stage='arrows',files=len(receipts))),flush=True)
    import shutil
    starts=[];total=0
    with (OUT/'arrow_tokens.u32').open('wb') as out:
        for r in receipts:
            root=OUT/'arrow_parts'/str(r['part'])
            for n,h in r['artifacts'].items():
                if sha(root/n)!=h: raise ValueError('Arrow part changed')
            starts.extend((np.load(root/'resets.npy')+total).tolist())
            with (root/'tokens.u32').open('rb') as src: shutil.copyfileobj(src,out,8*1024*1024)
            total+=r['tokens']
    if starts[0]!=0 or sum(r['rows'] for r in receipts)!=4716676: raise ValueError('Wrong Arrow coverage')
    tokens=np.memmap(OUT/'arrow_tokens.u32',mode='r',dtype='<u4');proofs=[]
    for a,b in zip(starts,starts[1:]+[total],strict=True):
        part=tokens[a:b]
        proofs.append(dict(tokens=b-a,sha256=hashlib.sha256(part.tobytes()).hexdigest(),prefix=part[:32].tolist()))
    dump(OUT/'ARROW_VERIFIED.json',dict(repository=TRAIN_REPO,revision=TRAIN_REVISION,
        rows=4716676,tokens=total,batches=len(proofs),proofs=proofs,files=receipts,
        tokens_sha256=sha(OUT/'arrow_tokens.u32')))

def align():
    raw=json.loads((OUT/'RAW_VERIFIED.json').read_text())
    arrow=json.loads((OUT/'ARROW_VERIFIED.json').read_text())
    for n,h in raw['artifacts'].items():
        if sha(OUT/n)!=h: raise ValueError('Raw token spool changed')
    index=np.load(OUT/'raw_index.npy',mmap_mode='r');tokens=np.memmap(OUT/'raw_tokens.u32',dtype='<u4',mode='r')
    order=np.random.default_rng(42).permutation(N);ntest=math.ceil(N*.002);train=order[ntest:]
    lengths=index[train,1].astype(np.int64)
    def fetch(a,b):
        return np.concatenate([tokens[int(index[r,0]):int(index[r,0]+index[r,1])] for r in train[a:b]]) if b>a else np.array([],dtype='<u4')
    started=time.time();keep=align_scope(lengths,fetch,arrow['proofs'])
    if int(keep.sum())!=arrow['tokens']: raise ValueError('Retained token coverage differs from training')
    coverage=np.zeros(N,dtype=np.uint32);coverage[train]=keep
    np.save(OUT/'coverage.npy',coverage)
    dump(OUT/'SCOPE_VERIFIED.json',dict(status='complete',all_arrow_batches_matched=True,
        source_sha256=raw['source_sha256'],raw_receipt_sha256=sha(OUT/'RAW_VERIFIED.json'),arrow_receipt_sha256=sha(OUT/'ARROW_VERIFIED.json'),
        raw_records=N,raw_train_records=len(train),heldout_records=ntest,
        retained_tokens=int(keep.sum()),dropped_training_tokens=int(lengths.sum()-keep.sum()),
        dropped_training_records=int((keep==0).sum()),partial_training_records=int(((keep>0)&(keep<lengths)).sum()),
        batches=arrow['batches'],coverage_sha256=sha(OUT/'coverage.npy'),elapsed_seconds=time.time()-started))

def convert():
    from transformers import AutoTokenizer
    scope=json.loads((OUT/'SCOPE_VERIFIED.json').read_text())
    if scope['status']!='complete' or not scope['all_arrow_batches_matched']: raise ValueError('Training scope not verified')
    if sha(SOURCE)!=scope['source_sha256'] or sha(OUT/'coverage.npy')!=scope['coverage_sha256']: raise ValueError('Source scope changed')
    if sha(OUT/'RAW_VERIFIED.json')!=scope['raw_receipt_sha256']: raise ValueError('Raw verification receipt changed')
    raw=json.loads((OUT/'RAW_VERIFIED.json').read_text())
    for n,h in raw['artifacts'].items():
        if sha(OUT/n)!=h: raise ValueError('Raw verified artifact changed '+n)
    coverage=np.load(OUT/'coverage.npy',mmap_mode='r');index=np.load(OUT/'raw_index.npy',mmap_mode='r')
    raw_tokens=np.memmap(OUT/'raw_tokens.u32',mode='r',dtype='<u4')
    tok=AutoTokenizer.from_pretrained(MODEL)
    stats=dict(partial_records=0,unicode_or_bpe_boundary_tokens_removed=0,body_words=0)
    def rows():
        with SOURCE.open() as src:
            for ordinal,line in enumerate(src):
                row=json.loads(line)
                if int(row['id'])!=ordinal: raise ValueError('Raw source order changed')
                keep=int(coverage[ordinal]);n=int(index[ordinal,1]);body=row['text']
                partial=0<keep<n
                if partial:
                    stats['partial_records']+=1
                    enc=tok(body,add_special_tokens=True,return_offsets_mapping=True,return_attention_mask=False)
                    ids=enc['input_ids']
                    if len(ids)!=n: raise ValueError('Tokenizer length changed')
                    off=int(index[ordinal,0])
                    if ids!=raw_tokens[off:off+n].tolist(): raise ValueError('Tokenizer tokens changed')
                    end=enc['offset_mapping'][keep-1][1] if keep>1 else 0
                    while end:
                        clipped=tok.encode(body[:end],add_special_tokens=True)
                        if len(clipped)<=keep and clipped==ids[:len(clipped)]: break
                        end-=1
                    body=body[:end];clipped=tok.encode(body,add_special_tokens=True)
                    stats['unicode_or_bpe_boundary_tokens_removed']+=keep-len(clipped)
                stats['body_words']+=len(body.split()) if keep else 0
                yield dict(id=row['id'],title=row['title'],text=body,keep=keep>0,gap_after=partial)
            if ordinal+1!=N: raise ValueError('Raw row count changed')
    corpus=OUT/'corpus.jsonl';partial=OUT/'corpus.jsonl.partial';count=0
    with partial.open('xb') as dst:
        for row in iter_chunks(rows()):
            dst.write((json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode())
            count+=1
            if count%1000000==0: print(json.dumps(dict(stage='convert',passages=count)),flush=True)
    partial.replace(corpus)
    dump(OUT/'CORPUS_VERIFIED.json',dict(status='complete',corpus_sha256=sha(corpus),rows=count,
        source_scope_sha256=sha(OUT/'SCOPE_VERIFIED.json'),source_sha256=scope['source_sha256'],
        segmentation='100 whitespace words, no overlap; join contiguous retained source rows within article; reset on source gaps and article boundary',
        retrieval_text='original title + newline + retained body passage',title_note='Title metadata retained as in previous RAG pipeline; memory trains body text only',statistics=stats))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0]);p.add_argument('stage',choices=['tokenize','arrows','align','convert'])
    p.add_argument('--source',type=Path,help='Atlas enwiki-dec2021 text-list-100-sec.jsonl');p.add_argument('--arrow-dir',type=Path,help='train split data-*.arrow files of '+TRAIN_REPO)
    p.add_argument('--tokenizer',default=MODEL);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    SOURCE,ARROWS,OUT,MODEL=a.source,a.arrow_dir,a.output,a.tokenizer
    os.environ['TOKENIZERS_PARALLELISM']='false'
    globals()[a.stage]()
