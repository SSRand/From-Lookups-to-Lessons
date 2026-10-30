"""Pure data contracts for a shared teacher cache."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def partition_rows(n,batch,workers,rank):
    if n<=0 or batch<=0 or workers<=0 or not 0<=rank<workers or n%batch:
        raise ValueError('Positive complete batches and valid worker rank required')
    return [j for i in range(rank*batch,n,batch*workers) for j in range(i,i+batch)]


def validate_row(row,target_tokens):
    if len(row['target_ids'])!=target_tokens:raise ValueError('Target count differs')
    for kind in ('student','teacher'):
        positions=row[kind+'_positions'];ids=row[kind+'_ids']
        if len(positions)!=target_tokens or positions!=list(range(positions[0],positions[0]+target_tokens)):
            raise ValueError('Noncontiguous or missing causal positions')
        if positions[0]<0 or positions[-1]+1>=len(ids):raise ValueError('Invalid causal offset')
        if [ids[x+1] for x in positions]!=row['target_ids']:raise ValueError('Shifted causal targets')
    docs=row['retrieved_documents']
    if len(docs)!=5 or any(d['title']==row['source_title'] for d in docs):
        raise ValueError('Teacher evidence contains own title or is incomplete')


def array_specs(n):
    return {'hidden':((n,4096),'uint16'),'base_logits':((n,32768),'uint16'),
            'teacher_ids':((n,64),'int32'),'teacher_probs':((n,64),'float32'),
            'gold':((n,),'int64')}
