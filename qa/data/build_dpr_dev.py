"""Build the DPR-train NQ/TriviaQA development pool, disjoint from every test question.

``calibration.jsonl`` holds 8,192 questions per dataset partitioned into
fit/dev/audit by a fixed hash; the tokenwise gate is fitted on ``fit``.
``dev.jsonl`` is the ``dev`` partition (4,091 questions), used to select the
single global coefficient of the correction/harm analysis and, after further
subsampling, the coefficients of the shared-teacher ablation.
"""
import ast
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import string
import argparse
import unicodedata
import urllib.request

SOURCES = {
    'nq': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv',
    'triviaqa': 'https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz',
}
PINNED = {'nq': 'e542ca0876138cdba632967cd8dede0bb719438b5b8b7a2c71d7eee069663190',
          'triviaqa': '400224bf4567b9355be819f26b47215ff3793827206157d190f76d26ad1816b1'}


def norm(q):
    q = unicodedata.normalize('NFKC', q).casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', q).split())


def digest(x):
    return hashlib.sha256(x.encode()).hexdigest()


TEST_DATASETS = ['nq', 'webq', 'triviaqa', 'popqa', 'hotpotqa']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-root', required=True, type=Path, help='directory with the frozen test <dataset>.jsonl files')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    data = args.output_dir; data.mkdir(parents=True, exist_ok=True)
    excluded = set(); evaluation_hashes = {}
    for name in TEST_DATASETS:
        path = args.test_root / f'{name}.jsonl'
        with path.open() as f:
            excluded.update(norm(json.loads(line)['question']) for line in f if line.strip())
        evaluation_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    seen = set(); manifest = {'evaluation_unique_questions': len(excluded), 'sources': {}, 'selection_seed': 'gate-search-v1', 'split_seed': 'gate-partition-v1'}
    all_rows = []
    for ds, url in SOURCES.items():
        path = data / url.rsplit('/', 1)[1]
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        raw = path.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        assert sha == PINNED[ds]
        text = gzip.decompress(raw).decode() if path.suffix == '.gz' else raw.decode()
        eligible = []; counters = {'source_rows': 0, 'evaluation_overlap': 0, 'duplicate': 0, 'empty': 0}
        for ordinal, row in enumerate(csv.reader(io.StringIO(text), delimiter='\t')):
            counters['source_rows'] += 1
            q, answers = row[0], ast.literal_eval(row[1])
            answers = list(dict.fromkeys(a for a in answers if isinstance(a, str) and a.strip()))
            key = norm(q)
            if not key or not answers:
                counters['empty'] += 1; continue
            if key in excluded:
                counters['evaluation_overlap'] += 1; continue
            if key in seen:
                counters['duplicate'] += 1; continue
            seen.add(key)
            bucket = int(digest('gate-partition-v1:' + key)[:8], 16) % 100
            split = 'fit' if bucket < 60 else 'dev' if bucket < 85 else 'audit'
            eligible.append({'dataset': ds, 'stable_id': f'dpr-{ds}-train-{ordinal}', 'question': q, 'aliases': answers, 'split': split, 'question_hash': digest(key), 'selection_hash': digest('gate-search-v1:' + key)})
        chosen = sorted(eligible, key=lambda x: x['selection_hash'])[:8192]
        # Fixed sorting and batches across all candidate methods.
        chosen.sort(key=lambda x: x['stable_id'])
        all_rows.extend(chosen)
        manifest['sources'][ds] = {'url': url, 'sha256': sha, 'bytes': len(raw), **counters, 'eligible': len(eligible), 'selected': len(chosen), 'splits': {s: sum(x['split'] == s for x in chosen) for s in ['fit', 'dev', 'audit']}}
    assert len({r['question_hash'] for r in all_rows}) == len(all_rows)
    target = data / 'calibration.jsonl'
    serialized = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows)
    if target.exists(): assert target.read_text() == serialized, 'refuse calibration drift'
    else: target.write_text(serialized)
    manifest['calibration_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    dev = data / 'dev.jsonl'
    dev.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows if r['split'] == 'dev'))
    manifest['dev_sha256'] = hashlib.sha256(dev.read_bytes()).hexdigest()
    manifest['evaluation_sha256'] = evaluation_hashes
    final = data / 'MANIFEST.json'
    if final.exists(): assert json.loads(final.read_text()) == manifest, 'refuse manifest drift'
    else: final.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
