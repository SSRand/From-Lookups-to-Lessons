"""Re-score the full-answer test grid into a paired bank; no selection on benchmark outcomes.

Inputs are qa.run_grid outputs under ``--runs``: ``base/`` (``--method base``)
and one directory per memory (``mlpmemory/``, ``resmem/``, ``memory_decoder/``)
generated at coefficients .1 .2 .3 .4 .5 .75 1. Coefficient 0 is the shared Base.

    python -m correction_harm.build_test_bank --test-root data/qa/test/frozen \
        --runs runs/fig4/test --output runs/fig4/bank
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from correction_harm.contract import LAMBDAS, METHODS, TEST_COUNTS, TEST_ORDER, question_key, transitions
from qa.common import cell_path, load_rows
from resmem.scoring import normalize_text, normalized_metrics, row_aliases


def md_extractor(base_model):
    """Memory Decoder answers are re-extracted from output_ids with the Base/MLP/ResMem first-line extractor."""
    from transformers import AutoTokenizer
    from qa.runtime import extract_first_answer_line
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return lambda ids: extract_first_answer_line(tokenizer.decode(ids, skip_special_tokens=True))


def score_arm(runs, method, lam, rows, reextract=None):
    """Per-row (EM, F1) of one grid cell, in the order of ``rows``."""
    folder = Path(runs) / ('base' if lam == 0 else method)
    idx = {r['stable_id']: i for i, r in enumerate(rows)}
    out = np.full((len(rows), 2), np.nan)
    for ds in dict.fromkeys(r['dataset'] for r in rows):
        for p in load_rows(cell_path(folder, ds, lam, 'predictions')):
            i = idx[p['stable_id']]
            assert rows[i]['dataset'] == ds and len(p['output_ids']) <= 12 and np.isnan(out[i, 0])
            text = reextract(p['output_ids']) if reextract is not None and method == 'memory_decoder' and lam != 0 else p['prediction']
            m = normalized_metrics(text, row_aliases(rows[i]))
            out[i] = m['exact_match'], m['token_f1']
    if np.isnan(out).any(): raise ValueError(f'incomplete predictions: {method} {lam}')
    return out


def load_test_rows(test_root):
    rows = [r for ds in TEST_ORDER for r in load_rows(Path(test_root) / f'{ds}.jsonl')]
    assert len({r['stable_id'] for r in rows}) == len(rows)
    assert Counter(r['dataset'] for r in rows) == TEST_COUNTS
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--test-root', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--base-model', default='mistralai/Mistral-7B-v0.3', help='tokenizer for Memory Decoder re-extraction')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    rows = load_test_rows(a.test_root)
    reextract = md_extractor(a.base_model) if 'memory_decoder' in a.methods else None
    base = score_arm(a.runs, 'base', 0., rows)[:, 0].astype(np.int8)
    em = np.full((len(a.methods), len(LAMBDAS), len(rows)), -1, dtype=np.int8)
    for mi, method in enumerate(a.methods):
        em[mi, 0] = base
        for li, lam in enumerate(LAMBDAS[1:], 1):
            em[mi, li] = score_arm(a.runs, method, lam, rows, reextract)[:, 0]
    assert np.isin(base, [0, 1]).all() and (em >= 0).all()
    a.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output/'scores.npz', base=base, em=em, methods=np.array(a.methods),
                        stable_id=np.array([r['stable_id'] for r in rows]),
                        dataset=np.array([r['dataset'] for r in rows]),
                        question_group=np.array([question_key(r['question']) for r in rows]))
    curves = []
    for dataset in ['pooled', 'nq', 'triviaqa', 'webq', 'hotpotqa', 'popqa']:
        mask = np.ones(len(rows), bool) if dataset == 'pooled' else np.array([r['dataset'] == dataset for r in rows])
        for mi, m in enumerate(a.methods):
            for li, l in enumerate(LAMBDAS):
                curves.append(dict(dataset=dataset, method=m, lambda_=l, **transitions(base[mask], em[mi, li, mask])))
    report = dict(rows=len(rows), methods=a.methods, base_correct=int(base.sum()), base_wrong=int((1-base).sum()),
                  question_clusters=len(set(question_key(r['question']) for r in rows)),
                  empty_normalized_alias_rows=[r['stable_id'] for r in rows if not any(normalize_text(x) for x in row_aliases(r))],
                  curves=curves, selection='None: all lambda points retained; independent dev result will choose operating points.')
    (a.output/'bank.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'curves'}, indent=2))


if __name__ == '__main__': main()
