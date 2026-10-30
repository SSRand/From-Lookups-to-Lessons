#!/usr/bin/env python
"""Build the BM25 retrieval index over English Wikipedia for the RAG teacher.
Pre-tokenized arrow shards (input_ids/labels, 2048-token chunks) -> detokenize the labeled (!=-100)
region of each chunk -> ONE passage per chunk (capped at 2000 chars) -> bm25s index.
Parallel detokenization via multiprocessing (workers write per-worker passage files), then a single bm25s index.
Usage: python -m teacher.build_bm25_index --data <arrow dir> --out <index dir> [--limit N] [--workers W]
"""
import argparse, glob, os, time
from multiprocessing import Process
from pathlib import Path
import pyarrow.ipc as ipc
import bm25s
from transformers import AutoTokenizer

def read_arrow(f):
    try: return ipc.open_file(f).read_all()
    except Exception: return ipc.open_stream(f).read_all()

def detok_worker(wid, files, tmp, base_model):
    tok=AutoTokenizer.from_pretrained(base_model)
    fo=open(os.path.join(tmp,f"psg_{wid}.txt"),"w"); n=0; t0=time.time()
    for k,f in enumerate(files):
        tb=read_arrow(f); ic=tb.column("input_ids").to_pylist(); lc=tb.column("labels").to_pylist()
        for ids,lab in zip(ic,lc):
            uniq=[t for t,l in zip(ids,lab) if l!=-100]
            if len(uniq)<16: continue
            # One passage per chunk (full labeled text, capped) keeps the index small and retrieval fast.
            # Retrieved text is truncated to PSG_CHARS at use time.
            txt=tok.decode(uniq, skip_special_tokens=True).strip().replace("\n"," ").replace("\t"," ")[:2000]
            if len(txt)>=200: fo.write(txt+"\n"); n+=1
        if (k+1)%20==0: print(f"[w{wid}] {k+1}/{len(files)} psg={n} {time.time()-t0:.0f}s", flush=True)
    fo.close(); print(f"[w{wid}] DONE psg={n} {time.time()-t0:.0f}s", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory containing train/*.arrow")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3", help="tokenizer used to pre-tokenize the shards")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=20)
    a=ap.parse_args()
    files=sorted(glob.glob(os.path.join(a.data,"train","*.arrow")))
    if a.limit>0: files=files[:a.limit]
    tmp=a.out+"_tmp"; Path(tmp).mkdir(parents=True, exist_ok=True)
    print(f"detok {len(files)} shards with {a.workers} workers", flush=True); t0=time.time()
    shards=[files[i::a.workers] for i in range(a.workers)]
    ps=[Process(target=detok_worker,args=(w,shards[w],tmp,a.base_model)) for w in range(a.workers) if shards[w]]
    for p in ps: p.start()
    for p in ps: p.join()
    print(f"detok done {time.time()-t0:.0f}s; loading passages", flush=True)
    corpus=[]
    for pf in sorted(glob.glob(os.path.join(tmp,"psg_*.txt"))):
        with open(pf) as fh: corpus.extend(l.rstrip("\n") for l in fh if l.strip())
    print(f"passages={len(corpus)}; indexing", flush=True)
    toks=bm25s.tokenize(corpus, stopwords="en", show_progress=True)
    retr=bm25s.BM25(corpus=corpus); retr.index(toks)
    retr.save(a.out, corpus=corpus)
    print(f"saved BM25 -> {a.out} total={time.time()-t0:.0f}s", flush=True)

if __name__=="__main__":
    main()
