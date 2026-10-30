"""Paired full-answer transitions and frozen independent-dev selection."""
import unicodedata
import re
import string
import numpy as np

LAMBDAS = (0., .1, .2, .3, .4, .5, .75, 1.)
METHODS = ('mlpmemory', 'resmem', 'memory_decoder')
TEST_ORDER = ('nq', 'triviaqa', 'webq', 'hotpotqa', 'popqa')
TEST_COUNTS = dict(nq=3610, triviaqa=11313, webq=2032, hotpotqa=7405, popqa=14267)
DEV_COUNTS = {'nq': 2050, 'triviaqa': 2041}

def question_key(s):
    return ' '.join(unicodedata.normalize('NFKC', s).casefold().split())

def strong_question_key(s):
    s = unicodedata.normalize('NFKC', s).casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def transitions(base, final):
    b, m = np.asarray(base), np.asarray(final)
    if b.ndim != 1 or b.shape != m.shape or not len(b) or not np.isin(b, [0, 1]).all() or not np.isin(m, [0, 1]).all():
        raise ValueError('Complete paired binary correctness required')
    b, m = b.astype(bool), m.astype(bool)
    n, bc = len(b), int(b.sum())
    r, h = int((~b & m).sum()), int((b & ~m).sum())
    return dict(n=n, base_correct=bc, base_wrong=n-bc, correction=r, harm=h,
                preserved=int((b & m).sum()), still_wrong=int((~b & ~m).sum()),
                correction_rate=r/(n-bc) if n != bc else None,
                harm_rate=h/bc if bc else None, correction_contribution=r/n,
                harm_contribution=h/n, net_em=(r-h)/n,
                base_em=bc/n, final_em=float(m.mean()))

def select_global(scores, datasets):
    datasets = np.asarray(datasets)
    if not len(datasets): raise ValueError('Empty dev population')
    def order(lam):
        a = np.asarray(scores[lam])
        if a.shape != (len(datasets), 2) or not np.isfinite(a).all():
            raise ValueError('Missing/nonfinite paired EM,F1')
        means = np.mean([a[datasets == d].mean(0) for d in sorted(set(datasets))], axis=0)
        return (-float(means[0]), -float(means[1]), float(lam))
    return min(scores, key=order)

def check_disjoint(dev, test):
    dk = [question_key(r['question']) for r in dev]
    if len(dk) != len(set(dk)): raise ValueError('Duplicate dev question')
    if len(dev) != len({r['stable_id'] for r in dev}): raise ValueError('Duplicate dev ID')
    if set(dk) & {question_key(r['question']) for r in test}: raise ValueError('Dev/test question overlap')

def validate_keys(keys, ids, methods=METHODS):
    keys, ids = list(keys), list(ids)
    if len(ids) != len(set(ids)): raise ValueError('Duplicate source ID')
    required = {(sid, 'base', 0.) for sid in ids}
    required |= {(sid, m, l) for sid in ids for m in methods for l in LAMBDAS[1:]}
    if len(keys) != len(set(keys)) or set(keys) != required:
        raise ValueError('Incomplete, duplicate, or extra predictions')
