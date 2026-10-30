"""Pure causal gate policy; no labels, question identities, or future tokens."""
import math

FEATURES = ['entropy', 'pmax', 'margin', 'fused_entropy', 'fused_pmax', 'fused_margin',
            'candidate_base_gap', 'residual_advantage', 'overshoot', 'candidate_logrank',
            'residual_std', 'entropy_change', 'lambda', 'position', 'first_entropy',
            'base_stop', 'res_stop', 'prefix_nonempty', 'prefix_words', 'prefix_chars',
            'prefix_numeric', 'question_words']
BASIC = ['entropy', 'pmax', 'margin', 'position', 'first_entropy', 'base_stop',
         'prefix_nonempty', 'prefix_words', 'prefix_numeric']
EXTRA_FEATURES = ['year_dash']


def feature_vector(f, feature_set):
    keys = BASIC if feature_set == 'basic' else FEATURES
    x = [float(f[k]) for k in keys]
    if feature_set == 'interactions':
        stop = f['base_stop'] * f['prefix_nonempty']
        x += [stop, stop * f['entropy'], stop * f['fused_margin'],
              stop * f['prefix_words'], f['entropy'] * f['fused_margin'],
              float(f['position'] == 1), float(f['position'] == 1) * f['entropy'],
              f['pmax'] * f['overshoot']]
    return x


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def validate_spec(s):
    def require(ok):
        if not ok: raise ValueError('invalid gate specification')
    def finite(x): return isinstance(x, (int, float)) and math.isfinite(x)
    k = s['kind']
    if k == 'constant': require(finite(s['value']) and 0 <= s['value'] <= 1)
    elif k == 'threshold': require(s['field'] in FEATURES + EXTRA_FEATURES and finite(s['threshold']) and s['op'] in ['ge', 'le'])
    elif k in ['all', 'any']:
        require(bool(s['rules']))
        for r in s['rules']: validate_spec(r)
    elif k == 'stop_guard':
        require(finite(s['threshold'])); validate_spec(s['otherwise'])
    elif k == 'position': validate_spec(s['first']); validate_spec(s['later'])
    elif k == 'hybrid': validate_spec(s['route']); validate_spec(s['guard'])
    elif k == 'soft_entropy':
        require(finite(s['threshold']) and finite(s['temperature']) and s['temperature'] > 0 and 0 <= s.get('floor', 0) <= 1)
    elif k == 'logistic':
        require(s['feature_set'] in ['basic', 'rich', 'interactions'])
        n = len(feature_vector({k: 0 for k in FEATURES}, s['feature_set']))
        for field in ['coef', 'mean', 'scale']:
            require(len(s[field]) == n and all(finite(v) for v in s[field]))
        require(all(v > 0 for v in s['scale']) and finite(s['intercept']) and finite(s['threshold']) and 0 <= s['threshold'] <= 1)
    else: raise ValueError(k)


def gate_value(f, spec):
    k = spec['kind']
    if k == 'constant':
        return float(spec['value'])
    if k == 'threshold':
        return float(f[spec['field']] >= spec['threshold'] if spec['op'] == 'ge' else f[spec['field']] <= spec['threshold'])
    if k == 'all':
        return min(gate_value(f, s) for s in spec['rules'])
    if k == 'any':
        return max(gate_value(f, s) for s in spec['rules'])
    if k == 'stop_guard':
        if f['base_stop'] and f['prefix_nonempty']:
            return float(f['entropy'] >= spec['threshold'])
        return gate_value(f, spec['otherwise'])
    if k == 'position':
        return gate_value(f, spec['first'] if f['position'] == 1 else spec['later'])
    if k == 'soft_entropy':
        return spec.get('floor', 0.0) + (1-spec.get('floor', 0.0)) * sigmoid((f['entropy']-spec['threshold'])/spec['temperature'])
    if k == 'logistic':
        x = feature_vector(f, spec['feature_set'])
        z = spec['intercept'] + sum(c*(v-m)/s for c, v, m, s in zip(spec['coef'], x, spec['mean'], spec['scale']))
        return float(sigmoid(z) >= spec['threshold'])
    raise ValueError(k)
