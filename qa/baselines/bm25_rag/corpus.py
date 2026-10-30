"""Source-scope alignment and article-bounded 100-word passage conversion."""
from __future__ import annotations
import hashlib
import numpy as np


def token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens,dtype='<u4').tobytes()).hexdigest()


def align_batches(rows, batches, block_size=2048):
    """Small in-memory adapter for the exact production alignment algorithm."""
    lengths=np.array([len(r) for r in rows],dtype=np.int64)
    offsets=np.r_[0,np.cumsum(lengths)]
    flat=np.array([x for r in rows for x in r],dtype='<u4')
    proofs=[dict(tokens=len(b),sha256=token_hash(b),prefix=list(b[:32])) for b in batches]
    return align_scope(lengths, lambda a,b: flat[offsets[a]:offsets[b]], proofs,block_size).tolist()


def align_scope(lengths, fetch, proofs, block_size=2048):
    """Align every retained batch with source-order train rows; fail closed."""
    lengths=np.asarray(lengths,dtype=np.int64)
    cumulative=np.r_[0,np.cumsum(lengths)]
    keep=np.zeros(len(lengths),dtype=np.uint32)
    start=0
    for i, proof in enumerate(proofs):
        k=int(proof['tokens'])
        if k<=0: raise ValueError('Empty training token batch')
        end_token=int(cumulative[start])+k
        full_end=int(np.searchsorted(cumulative,end_token,side='right')-1)
        through=full_end+(int(cumulative[full_end])<end_token)
        actual=fetch(start,through)[:k]
        if len(actual)!=k or token_hash(actual)!=proof['sha256']:
            raise ValueError(f'Training token mismatch at batch {i}, raw train position {start}')
        keep[start:full_end]=lengths[start:full_end]
        if through>full_end: keep[full_end]=end_token-int(cumulative[full_end])
        # The original map floors at block_size, except a smaller complete batch.
        low=int(np.searchsorted(cumulative,end_token,side='left'))
        high=int(np.searchsorted(cumulative,end_token+(block_size if k>=block_size else 1),side='left'))
        candidates=[]
        for stop in range(max(start+1,low),min(len(lengths)+1,high)):
            total=int(cumulative[stop]-cumulative[start])
            retained=(total//block_size)*block_size if total>=block_size else total
            if retained!=k: continue
            if i+1==len(proofs):
                if stop==len(lengths): candidates.append(stop)
                continue
            nxt=proofs[i+1]; nextk=int(nxt['tokens'])
            nextend=int(np.searchsorted(cumulative,int(cumulative[stop])+nextk,side='left'))
            if nextend>len(lengths): continue
            # Full next-batch verification also disambiguates duplicate bodies.
            prefix_end=int(np.searchsorted(cumulative,int(cumulative[stop])+len(nxt['prefix']),side='left'))
            if prefix_end>len(lengths): continue
            prefix=fetch(stop,prefix_end)[:len(nxt['prefix'])]
            if list(prefix)!=nxt['prefix']: continue
            tokens=fetch(stop,nextend)[:nextk]
            if len(tokens)==nextk and token_hash(tokens)==nxt['sha256']:
                candidates.append(stop)
        if len(candidates)!=1:
            raise ValueError(f'Ambiguous/unmatched training token boundary at batch {i}: {candidates}')
        start=candidates[0]
    if start!=len(lengths): raise ValueError('Unaccounted raw training records')
    return keep


def iter_chunks(records, words=100):
    """Keep source order, join adjacent body spans, reset at any corpus gap."""
    pending=[]; title=None; first=None; ordinal=0; previous=-1
    def emit():
        nonlocal pending,ordinal
        passage=' '.join(pending); pending=[]
        row=dict(id=str(ordinal),title=title,passage=passage,text=title+'\n'+passage)
        ordinal+=1
        return row
    for row in records:
        rid=int(row['id'])
        if rid<=previous: raise ValueError('Source IDs out of order')
        if (rid!=previous+1 or not row['keep'] or row['title']!=title) and pending:
            yield emit()
        previous=rid
        if not row['keep']: title=None; continue
        title=row['title'].strip()
        if not title: raise ValueError('Source title missing')
        for word in row['text'].split():
            pending.append(word)
            if len(pending)==words: yield emit()
        if row.get('gap_after'):
            if pending: yield emit()
            title=None
    if pending: yield emit()
