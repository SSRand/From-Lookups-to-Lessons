"""Teacher stage 1 (CPU): BM25 retrieval + context assembly for the RAG teacher cache.
For each 64-token span (over the labeled region of each 2048-token chunk) it BM25-retrieves the top-3
passages using the first 48 span tokens as the query and builds the full (context + span) token sequence.
Stores prepped npz {full_ids, clens, span_lens, span_ids} per output chunk. No GPU / no torch; run with
many cores (multiprocessing over shards). teacher.forward_teacher then GPU-forwards these to
softmax(z_ctx) top-64. Resumable per (job, worker).
"""
import os
# One BLAS thread per worker: otherwise each of N worker processes spawns ~ncores OMP/BLAS threads
# (numpy/scipy in bm25s) and oversubscribes the machine. Set BEFORE the numpy import.
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v,"1")
import argparse, glob, time
from multiprocessing import Process
from pathlib import Path
import numpy as np
import pyarrow.ipc as ipc
import json
from transformers import AutoTokenizer
import bm25s
SPAN_TOK=64; TOPK_RET=3; PSG_CHARS=450; MAX_CTX=1024; SPANS_PER_CHUNK=20000

def read_arrow(f):
    try: return ipc.open_file(f).read_all()
    except Exception: return ipc.open_stream(f).read_all()

def retrieve_batch(retr, texts):
    out=[[] for _ in texts]
    nz=[i for i,t in enumerate(texts) if t and t.strip()]
    if not nz: return out
    try:
        qs=bm25s.tokenize([texts[i][:600] for i in nz], stopwords="en", show_progress=False)
        res,_=retr.retrieve(qs, k=TOPK_RET, show_progress=False)
        for r,i in enumerate(nz):
            out[i]=[str(res[r,j]["text"])[:PSG_CHARS] for j in range(res.shape[1])]
    except Exception: pass
    return out


def _resolve_input_files(*, data_root: str, shard_manifest: str, max_shards: int, job_id: int, njobs: int) -> list[str]:
    if shard_manifest:
        payload = json.loads(Path(shard_manifest).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            files = payload.get("selected_shards")
        else:
            files = payload
        if not isinstance(files, list) or not all(isinstance(item, str) for item in files):
            raise ValueError("shard_manifest must be a JSON list of shard paths or an object with selected_shards")
        resolved = sorted(files)
    else:
        resolved = sorted(glob.glob(os.path.join(data_root, "train", "*.arrow")))
    if max_shards > 0:
        resolved = resolved[:max_shards]
    return resolved[job_id::njobs]

def worker(wid, nworkers, args):
    """Cut 64-token spans, retrieve the top-3 BM25 passages with the first 48 span tokens as the query,
    and write the (context + span) token sequences."""
    tok=AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    retr=bm25s.BM25.load(args.bm25, load_corpus=True, mmap=True)
    files=_resolve_input_files(
        data_root=args.data,
        shard_manifest=args.shard_manifest,
        max_shards=args.max_shards,
        job_id=args.job_id,
        njobs=args.njobs,
    )
    my_files=[files[i] for i in range(wid, len(files), nworkers)]
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    B=int(args.batch_size); t0=time.time()
    print(f"[w{wid}] {len(my_files)} shards", flush=True)
    def span_gen():
        for f in my_files:
            tb=read_arrow(f); ic=tb.column("input_ids").to_pylist(); lc=tb.column("labels").to_pylist()
            for ids,lab in zip(ic,lc):
                uniq=[t for t,l in zip(ids,lab) if l!=-100]
                for s in range(0,len(uniq)-1,SPAN_TOK):
                    ch=uniq[s:s+SPAN_TOK]
                    if len(ch)>=8: yield ch
    fid_buf=[]; cl_buf=[]; sl_buf=[]; sid_buf=[]; chunk=0
    exist=sorted(glob.glob(str(out/f"w{args.job_id}_{wid}_chunk*.npz"))); skip=0
    for ef in exist:
        try: skip+=int(np.load(ef,mmap_mode="r")["span_lens"].shape[0])
        except Exception: pass
    chunk=len(exist)
    def flush():
        nonlocal chunk
        if not fid_buf: return
        maxF=max(len(f) for f in fid_buf); maxS=max(len(s) for s in sid_buf)
        F=np.full((len(fid_buf),maxF), tok.pad_token_id, np.int32)
        S=np.full((len(sid_buf),maxS), tok.pad_token_id, np.int32)
        for i,f in enumerate(fid_buf): F[i,:len(f)]=f
        for i,s in enumerate(sid_buf): S[i,:len(s)]=s
        np.savez(out/f"w{args.job_id}_{wid}_chunk{chunk:04d}.npz", full_ids=F, clens=np.array(cl_buf,np.int32),
                 span_lens=np.array(sl_buf,np.int32), span_ids=S)
        print(f"[w{wid}] wrote chunk{chunk} n={len(fid_buf)} {time.time()-t0:.0f}s", flush=True)
        chunk+=1; fid_buf.clear(); cl_buf.clear(); sl_buf.clear(); sid_buf.clear()
    def do(batch):
        psgs=retrieve_batch(retr, tok.batch_decode([x[:48] for x in batch], skip_special_tokens=True))
        for i,sp in enumerate(batch):
            Li=len(sp); ctx="Context:\n"+"\n".join(psgs[i])+"\n\n"
            cids=tok(ctx, truncation=True, max_length=MAX_CTX-Li).input_ids
            fid_buf.append(np.array(cids+list(sp),np.int32)); cl_buf.append(len(cids))
            sl_buf.append(Li); sid_buf.append(np.array(sp,np.int32))
    idx=0; buf=[]; nb=0
    for sp in span_gen():
        if idx<skip: idx+=1; continue
        idx+=1
        if args.max_spans>0 and (idx-skip)>args.max_spans: break
        buf.append(sp)
        if len(buf)<B: continue
        do(buf); buf=[]; nb+=1
        if nb%50==0:
            print(f"[w{wid}] nb={nb} spans={idx-skip} ({(idx-skip)/max(time.time()-t0,1):.0f} sp/s) {time.time()-t0:.0f}s", flush=True)
        if len(fid_buf)>=SPANS_PER_CHUNK: flush()
    if buf: do(buf)
    if fid_buf: flush()
    print(f"[w{wid}] DONE spans={idx-skip} {time.time()-t0:.0f}s", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bm25", required=True, help="index directory written by teacher.build_bm25_index")
    ap.add_argument("--data", default="", help="directory containing train/*.arrow (ignored with --shard_manifest)")
    ap.add_argument("--shard_manifest", default="", help="optional JSON list of shard paths (or {selected_shards: [...]})")
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_spans", type=int, default=0); ap.add_argument("--max_shards", type=int, default=0)
    ap.add_argument("--njobs", type=int, default=1); ap.add_argument("--job_id", type=int, default=0)
    ap.add_argument("--local_index", type=int, default=1)
    a=ap.parse_args()
    if not a.data and not a.shard_manifest:
        ap.error("one of --data or --shard_manifest is required")
    # copy the BM25 index to node-local /dev/shm once (avoids network-filesystem mmap IO under many workers)
    if a.local_index:
        import shutil
        src=a.bm25; local="/dev/shm/bm25_local"
        if not os.path.exists(local):
            print(f"copying BM25 index {src} -> {local} (node-local)...", flush=True); t=time.time()
            try:
                shutil.copytree(src, local); print(f"copied {time.time()-t:.0f}s", flush=True); a.bm25=local
            except Exception as e:
                print(f"local copy failed ({e}); using mmap from {src}", flush=True)
        else:
            a.bm25=local; print("local index present, reuse", flush=True)
    if a.workers==1: worker(0,1,a); return
    ps=[Process(target=worker,args=(w,a.workers,a)) for w in range(a.workers)]
    for p in ps: p.start()
    for p in ps: p.join()

if __name__=="__main__":
    main()
