"""Frozen independent-dev selection of one global coefficient per method.

Dev generations are qa.run_grid outputs on ``dev.jsonl`` (from
qa.data.build_dpr_dev), laid out as for build_test_bank:

    python -m qa.run_grid --rows data/qa/dpr_dev/dev.jsonl --method base --coefficients 0 --output runs/fig4/dev/base
    python -m qa.run_grid --rows data/qa/dpr_dev/dev.jsonl --method resmem --checkpoint CKPT \
        --coefficients 0.1 0.2 0.3 0.4 0.5 0.75 1 --output runs/fig4/dev/resmem
    python -m correction_harm.select_dev --dev data/qa/dpr_dev/dev.jsonl --test-root data/qa/test/frozen \
        --runs runs/fig4/dev --output runs/fig4/SELECTION.json

Selection maximizes equal-weight NQ/TriviaQA dev EM, then F1, then prefers the smaller coefficient.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from correction_harm.build_test_bank import load_test_rows, md_extractor, score_arm
from correction_harm.contract import DEV_COUNTS, LAMBDAS, METHODS, check_disjoint, select_global, strong_question_key
from qa.common import load_rows

RULE = 'One global scalar per method; equal-NQ/TriviaQA dev EM, then F1, then smaller coefficient, over 0/.1/.2/.3/.4/.5/.75/1.'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dev', type=Path, required=True)
    ap.add_argument('--test-root', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3', help='tokenizer for Memory Decoder re-extraction')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    data = load_rows(a.dev)
    if Counter(r['dataset'] for r in data) != DEV_COUNTS: raise ValueError('dev coverage')
    test = load_test_rows(a.test_root)
    check_disjoint(data, test)
    if {strong_question_key(r['question']) for r in data} & {strong_question_key(r['question']) for r in test}:
        raise ValueError('strong-normalized dev/test overlap')
    reextract = md_extractor(a.base_model) if 'memory_decoder' in a.methods else None
    ds = np.array([r['dataset'] for r in data])
    base = score_arm(a.runs, 'base', 0., data)
    all_scores, selected = {}, {}
    for method in a.methods:
        scores = {l: base if l == 0 else score_arm(a.runs, method, l, data, reextract) for l in LAMBDAS}
        selected[method] = float(select_global(scores, ds))
        all_scores[method] = {str(l): dict(per_dataset={d: dict(zip(('em', 'f1'), s[ds == d].mean(0).tolist())) for d in sorted(set(ds))},
                                           macro=dict(zip(('em', 'f1'), np.mean([s[ds == d].mean(0) for d in sorted(set(ds))], 0).tolist())))
                              for l, s in scores.items()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(status='complete', selected=selected, scores=all_scores, selection=RULE, n=len(data),
                                        counts=DEV_COUNTS, frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())), indent=2)+'\n')
    print(json.dumps(selected))


if __name__ == '__main__': main()
