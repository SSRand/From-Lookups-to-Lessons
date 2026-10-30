"""Teacher stage 2 (GPU): forward the prepped (context + span) sequences from teacher.prepare_contexts
to the RAG teacher distribution. Per span: base forward over full_ids -> softmax(z_ctx) at span positions
-> top-64 (teacher_tok/teacher_prob). Stores the cached targets {span_ids, span_lens, teacher_tok, teacher_prob} consumed
by train.pretrain. No retrieval in the loop, so the GPU stays saturated. mp.spawn over GPUs; resumable per rank.
Length-sorted batching (less padding) + vectorized lm_head/softmax/topk over all span positions in a batch
(one kernel + one D2H copy). z_base is recomputed at train time.
"""
import argparse, sys, os, glob, time
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM
SPANS_PER_CHUNK=20000

def order_rows(rows, preserve_order=False):
    if preserve_order:
        return rows
    return sorted(rows, key=lambda r: len(r[0]))

def worker(rank, world, args):
    """Run the frozen base on (context + span) and cache the top-64 teacher distribution
    at every span position."""
    torch.cuda.set_device(rank); dev=f"cuda:{rank}"
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    K=int(args.topk); B=int(args.batch_size)
    time.sleep(rank*int(args.load_stagger))
    tok=AutoTokenizer.from_pretrained(args.base_model); pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    base=AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                              low_cpu_mem_usage=True).cuda(rank).eval()
    normcap={"x":None}
    base.model.norm.register_forward_hook(lambda _m,_i,o: normcap.__setitem__("x", o.detach()))
    pfiles=sorted(glob.glob(os.path.join(args.prepped,"w*_chunk*.npz")))
    pfiles=pfiles[args.job_id::args.njobs]        # job-level shard (disjoint across parallel jobs)
    my=[pfiles[i] for i in range(rank,len(pfiles),world)]
    print(f"[rank{rank}] {len(my)} prepped files", flush=True)
    sid_buf=[]; len_buf=[]; ctok_buf=[]; cprob_buf=[]; ochunk=0; t0=time.time(); done=0
    ochunk=len(sorted(glob.glob(str(out/f"rank{args.job_id}_{rank}_chunk*.npz"))))
    def flush():
        nonlocal ochunk
        if not sid_buf: return
        maxL=max(len(s) for s in sid_buf)
        S=np.full((len(sid_buf),maxL), pad, np.int32)
        for i,s in enumerate(sid_buf): S[i,:len(s)]=s
        np.savez(out/f"rank{args.job_id}_{rank}_chunk{ochunk:04d}.npz", span_ids=S, span_lens=np.array(len_buf,np.int32),
                 teacher_tok=np.concatenate(ctok_buf,0), teacher_prob=np.concatenate(cprob_buf,0))
        print(f"[rank{rank}] wrote chunk{ochunk} spans={len(sid_buf)} {time.time()-t0:.0f}s", flush=True)
        ochunk+=1; sid_buf.clear(); len_buf.clear(); ctok_buf.clear(); cprob_buf.clear()

    @torch.no_grad()
    def proc(rows):
        maxF=max(len(r[0]) for r in rows)
        fids=torch.full((len(rows),maxF), pad, dtype=torch.long, device=dev)
        fmask=torch.zeros((len(rows),maxF), dtype=torch.long, device=dev)
        for i,r in enumerate(rows):
            L=len(r[0]); fids[i,:L]=torch.tensor(r[0],device=dev); fmask[i,:L]=1
        base(input_ids=fids, attention_mask=fmask)
        znorm=normcap["x"]                                   # [B,maxF,H] bf16
        # gather all span positions (clen .. clen+Li-1) across the batch -> one flat tensor
        hs=[]; bounds=[]
        for i,r in enumerate(rows):
            clen=int(r[1]); Li=int(r[2])
            if Li<8: bounds.append(0); continue
            hs.append(znorm[i, clen:clen+Li-1]); bounds.append(Li-1)
        if not hs: return
        H=torch.cat(hs,0)                                    # [totalpos,H]
        Z=base.lm_head(H).float()                            # [totalpos,V]  single kernel
        cpv,cpi=torch.softmax(Z,dim=-1).topk(K,dim=-1)       # [totalpos,K]
        cpv=cpv.to("cpu").numpy().astype(np.float16); cpi=cpi.to("cpu").numpy().astype(np.int32)  # one D2H
        off=0
        for i,r in enumerate(rows):
            b=bounds[i]
            if b==0: continue
            sid_buf.append(r[3].astype(np.int32)); len_buf.append(int(r[2]))
            ctok_buf.append(cpi[off:off+b]); cprob_buf.append(cpv[off:off+b]); off+=b

    # load all rows for this rank, sort by full length for tight batches
    rows=[]
    for pf in my:
        d=np.load(pf); F=d["full_ids"]; CL=d["clens"]; SL=d["span_lens"]; SID=d["span_ids"]
        for i in range(F.shape[0]):
            cl=int(CL[i]); Li=int(SL[i])
            rows.append((F[i,:cl+Li], cl, Li, SID[i,:Li]))
        del d
        if args.max_spans>0 and len(rows)>=args.max_spans*1.2: break
    rows=order_rows(rows, preserve_order=args.preserve_order)
    print(f"[rank{rank}] {len(rows)} spans loaded+sorted; forwarding B={B}", flush=True)
    batch=[]
    for r in rows:
        batch.append(r)
        if len(batch)>=B:
            proc(batch); done+=len(batch); batch=[]
            if (done//B)%25==0:
                print(f"[rank{rank}] done={done} ({done/max(time.time()-t0,1):.0f} sp/s) {time.time()-t0:.0f}s", flush=True)
            if len(sid_buf)>=SPANS_PER_CHUNK: flush()
            if args.max_spans>0 and done>=args.max_spans: break
    if batch and not (args.max_spans>0 and done>=args.max_spans): proc(batch); done+=len(batch)
    if sid_buf: flush()
    print(f"[rank{rank}] DONE spans={done} {time.time()-t0:.0f}s", flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prepped", required=True, help="output directory of teacher.prepare_contexts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-model", default="mistralai/Mistral-7B-v0.3")
    ap.add_argument("--topk", type=int, default=64); ap.add_argument("--batch_size", type=int, default=192)
    ap.add_argument("--world", type=int, default=8); ap.add_argument("--load_stagger", type=int, default=15)
    ap.add_argument("--max_spans", type=int, default=0)
    ap.add_argument("--njobs", type=int, default=1); ap.add_argument("--job_id", type=int, default=0)
    ap.add_argument("--preserve_order", action="store_true")
    a=ap.parse_args()
    if a.world==1: worker(0,1,a)
    else: mp.spawn(worker, args=(a.world,a), nprocs=a.world, join=True)

if __name__=="__main__":
    try: main()
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(1)
