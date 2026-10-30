"""Question-only BM25 retrieval of the top-20 passages for every test question (CPU only).

    python -m qa.baselines.bm25_rag.retrieve --test-root data/qa/test/frozen \
        --corpus CORPUS_DIR/corpus.jsonl --index CORPUS_DIR/bm25s-index --output runs/bm25_rag/retrieval
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import numpy as np

from qa.baselines.bm25_rag.index import validate_bm25_index
from qa.common import DATASETS

TOKEN_PATTERN_RE = re.compile(r"(?u)\b\w\w+\b")


@dataclass(frozen=True)
class Document:
    doc_id: str | int
    text: str


@dataclass(frozen=True)
class SparseBM25Index:
    vocab: Mapping[str, int]
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    corpus_path: Path
    corpus_offsets: np.ndarray
    num_docs: int


@lru_cache(maxsize=4)
def _load_bm25s_index(index_path: str) -> SparseBM25Index:
    root = Path(index_path)
    params = json.loads((root / "params.index.json").read_text(encoding="utf-8"))
    vocab = json.loads((root / "vocab.index.json").read_text(encoding="utf-8"))
    corpus_path = root / "corpus.jsonl"
    mmindex_path = root / "corpus.mmindex.json"
    if mmindex_path.exists():
        raw_offsets = json.loads(mmindex_path.read_text(encoding="utf-8"))
        if not isinstance(raw_offsets, list):
            raise ValueError(f"BM25s corpus mmindex must be a JSON array: {mmindex_path}")
        corpus_offsets = np.asarray(raw_offsets, dtype=np.int64)
    else:
        offsets: list[int] = []
        with corpus_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                if not handle.readline():
                    break
                offsets.append(offset)
        corpus_offsets = np.asarray(offsets, dtype=np.int64)
    if corpus_offsets.ndim != 1 or np.any(corpus_offsets < 0):
        raise ValueError(f"BM25s corpus mmindex contains invalid offsets: {mmindex_path}")
    if corpus_offsets.size > 1 and np.any(corpus_offsets[1:] <= corpus_offsets[:-1]):
        raise ValueError(f"BM25s corpus mmindex offsets must be strictly increasing: {mmindex_path}")
    return SparseBM25Index(
        vocab={str(token): int(token_id) for token, token_id in vocab.items()},
        data=np.load(root / "data.csc.index.npy", mmap_mode="r"),
        indices=np.load(root / "indices.csc.index.npy", mmap_mode="r"),
        indptr=np.load(root / "indptr.csc.index.npy", mmap_mode="r"),
        corpus_path=corpus_path,
        corpus_offsets=corpus_offsets,
        num_docs=int(params["num_docs"]),
    )


def _tokenize_bm25s_query(text: str) -> list[str]:
    return TOKEN_PATTERN_RE.findall(text.lower())


def _load_bm25s_documents(retriever: SparseBM25Index, doc_indices: Sequence[int]) -> list[Document]:
    documents: list[Document] = []
    with retriever.corpus_path.open("rb") as handle:
        for doc_index in doc_indices:
            handle.seek(int(retriever.corpus_offsets[doc_index]))
            line = handle.readline()
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"BM25s corpus row {doc_index} is not valid JSON") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"BM25s corpus row {doc_index} must be an object")
            doc_text = str(row.get("text") or row.get("passage") or "").strip()
            if not doc_text:
                raise ValueError(f"BM25s corpus row {doc_index} requires non-empty text or passage")
            doc_id = str(row.get("id") or row.get("_id") or doc_index)
            documents.append(Document(doc_id=doc_id, text=doc_text))
    return documents


def rank_documents_bm25s(query: str, *, index_path: str, top_k: int) -> list[Document]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    retriever = _load_bm25s_index(index_path)
    token_ids: list[int] = []
    for token in _tokenize_bm25s_query(query[:600]):
        if token not in retriever.vocab:
            continue
        token_id = retriever.vocab[token]
        token_ids.append(token_id)
    if not token_ids or retriever.num_docs <= 0 or retriever.corpus_offsets.size == 0:
        return []
    posting_doc_ids_by_token: dict[int, np.ndarray] = {}
    posting_scores_by_token: dict[int, np.ndarray] = {}
    corpus_size = min(retriever.num_docs, int(retriever.corpus_offsets.size))
    for token_id in dict.fromkeys(token_ids):
        if token_id < 0 or token_id + 1 >= retriever.indptr.shape[0]:
            continue
        start = int(retriever.indptr[token_id])
        end = int(retriever.indptr[token_id + 1])
        if end <= start:
            continue
        doc_ids = np.asarray(retriever.indices[start:end])
        scores = np.asarray(retriever.data[start:end], dtype=np.float32)
        valid = (doc_ids >= 0) & (doc_ids < corpus_size) & np.isfinite(scores)
        if np.any(valid):
            posting_doc_ids_by_token[token_id] = doc_ids[valid].astype(np.int64, copy=False)
            posting_scores_by_token[token_id] = scores[valid]
    if not posting_doc_ids_by_token:
        return []
    candidate_doc_ids = np.unique(np.concatenate(list(posting_doc_ids_by_token.values())))
    candidate_scores = np.zeros(candidate_doc_ids.shape[0], dtype=np.float32)
    posting_inverse_by_token = {
        token_id: np.searchsorted(candidate_doc_ids, doc_ids)
        for token_id, doc_ids in posting_doc_ids_by_token.items()
    }
    for token_id in token_ids:
        if token_id not in posting_inverse_by_token:
            continue
        np.add.at(candidate_scores, posting_inverse_by_token[token_id], posting_scores_by_token[token_id])
    positive = np.isfinite(candidate_scores) & (candidate_scores > 0.0)
    candidate_doc_ids = candidate_doc_ids[positive]
    candidate_scores = candidate_scores[positive]
    if candidate_doc_ids.size == 0:
        return []

    limit = min(top_k, int(candidate_doc_ids.size))
    if candidate_doc_ids.size > limit:
        cutoff = float(np.partition(candidate_scores, candidate_scores.size - limit)[candidate_scores.size - limit])
        selected = np.flatnonzero(candidate_scores > cutoff)
        tied = np.flatnonzero(candidate_scores == cutoff)
        selected = np.concatenate((selected, tied[: limit - selected.size]))
    else:
        selected = np.arange(candidate_doc_ids.size)
    selected_doc_ids = candidate_doc_ids[selected]
    selected_scores = candidate_scores[selected]
    order = np.lexsort((selected_doc_ids, -selected_scores))
    ranked_doc_ids = selected_doc_ids[order].tolist()
    return _load_bm25s_documents(retriever, ranked_doc_ids)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--test-root', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--chunk-size', type=int, default=64)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Verify every published index artifact once, before any parallel readers.
    manifest = validate_bm25_index(args.corpus, args.index)
    index = str(args.index)
    loaded = _load_bm25s_index(index)
    assert loaded.num_docs == manifest['corpus']['row_count']
    assert len(loaded.corpus_offsets) == manifest['corpus']['row_count']
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    started = time.monotonic()
    total = 0
    def retrieve(row):
        docs = rank_documents_bm25s(row['question'], index_path=index, top_k=args.top_k)
        return {'stable_id': row['stable_id'], 'dataset': row['dataset'],
                'question': row['question'], 'retrieval_method': 'bm25s',
                'retrieved_documents': [{'rank': i, 'doc_id': d.doc_id, 'text': d.text}
                                         for i, d in enumerate(docs, start=1)]}
    for dataset in DATASETS:
        with (args.test_root / f'{dataset}.jsonl').open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        ids = [r['stable_id'] for r in rows]
        assert len(set(ids)) == len(ids)
        path = output / (dataset + '__bm25_rag.retrieval.jsonl')
        partial = path.with_suffix(path.suffix + '.partial')
        completed = []
        with partial.open('x') as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(rows), args.chunk_size):
                batch = rows[start:start + args.chunk_size]
                for result in pool.map(retrieve, batch):
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + '\n')
                    completed.append(result['stable_id'])
                handle.flush()
                print(json.dumps({'dataset': dataset, 'completed_rows': len(completed),
                    'total_rows': len(rows), 'elapsed_seconds': time.monotonic()-started}), flush=True)
        assert completed == ids
        partial.rename(path)
        artifacts[path.name] = {'sha256': sha(path), 'rows': len(rows), 'bytes': path.stat().st_size}
        total += len(rows)
    receipt = {'status': 'complete', 'top_k': args.top_k, 'index_manifest_sha256': sha(args.index / 'bm25_manifest.json'),
               'corpus_sha256': manifest['corpus']['sha256'], 'row_count': total,
               'runtime_seconds': time.monotonic()-started, 'artifacts': artifacts,
               'source_answer_aliases': 'Unmodified original dataset remains authoritative for scoring; retrieval uses question only.'}
    (output/'COMPLETION.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'status': 'complete', 'rows': total}))


if __name__ == '__main__':
    main()
