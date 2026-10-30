import unittest
from gate.gates import gate_value, feature_vector, validate_spec, FEATURES


class GateTests(unittest.TestCase):
    def setUp(self):
        self.f = {k: 0.0 for k in FEATURES}
        self.f.update(entropy=1.0, pmax=0.7, margin=2.0, base_stop=1.0, prefix_nonempty=1.0)

    def test_endpoints(self):
        self.assertEqual(gate_value(self.f, {'kind': 'constant', 'value': 0}), 0)
        self.assertEqual(gate_value(self.f, {'kind': 'constant', 'value': 1}), 1)

    def test_combination(self):
        high = {'kind': 'threshold', 'field': 'entropy', 'threshold': 2, 'op': 'ge'}
        low = {'kind': 'threshold', 'field': 'margin', 'threshold': 3, 'op': 'le'}
        self.assertEqual(gate_value(self.f, {'kind': 'all', 'rules': [high, low]}), 0)
        self.assertEqual(gate_value(self.f, {'kind': 'any', 'rules': [high, low]}), 1)

    def test_stop_guard_distinguishes_blank(self):
        s = {'kind': 'stop_guard', 'threshold': 2, 'otherwise': {'kind': 'constant', 'value': 1}}
        self.assertEqual(gate_value(self.f, s), 0)
        self.f['prefix_nonempty'] = 0
        self.assertEqual(gate_value(self.f, s), 1)

    def test_no_gold_or_future_inputs(self):
        before = feature_vector(self.f, 'interactions')
        self.f.update(aliases=['SECRET'], base_em=1, res_em=0, stable_id='ID', future='SECRET')
        self.assertEqual(before, feature_vector(self.f, 'interactions'))

    def test_soft_bounds_and_monotonicity(self):
        s = {'kind': 'soft_entropy', 'threshold': 2, 'temperature': 0.5, 'floor': 0.1}
        a = gate_value(self.f, s)
        self.f['entropy'] = 4
        b = gate_value(self.f, s)
        self.assertTrue(0.1 <= a < b <= 1)

    def test_logistic_transform(self):
        n = len(feature_vector(self.f, 'basic'))
        s = {'kind': 'logistic', 'feature_set': 'basic', 'mean': [0]*n, 'scale': [1]*n, 'coef': [0]*n, 'intercept': 0, 'threshold': 0.5}
        self.assertEqual(gate_value(self.f, s), 1)

    def test_reject_invalid_model_dimensions_and_weights(self):
        with self.assertRaises(ValueError):
            validate_spec({'kind': 'logistic', 'feature_set': 'basic', 'mean': [0], 'scale': [1], 'coef': [0], 'intercept': 0, 'threshold': .5})
        for spec in [{'kind': 'constant', 'value': 1.2}, {'kind': 'soft_entropy', 'threshold': 1, 'temperature': 0}]:
            with self.assertRaises(ValueError): validate_spec(spec)


if __name__ == '__main__':
    unittest.main()
