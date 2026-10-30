import unittest
import numpy as np
from correction_harm import contract


class Contracts(unittest.TestCase):
    def test_four_transitions_reconcile_net_and_distinct_denominators(self):
        # Two repairs, one harm, one preserved and two still wrong.
        v = contract.transitions([0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0])
        self.assertEqual([v[k] for k in ('correction', 'harm', 'preserved', 'still_wrong')], [2, 1, 1, 2])
        self.assertAlmostEqual(v['correction_rate'], .5)
        self.assertAlmostEqual(v['harm_rate'], .5)
        self.assertAlmostEqual(v['net_em'], 1 / 6)
        self.assertAlmostEqual(v['correction_contribution'] - v['harm_contribution'], v['net_em'])

    def test_dev_selection_is_equal_dataset_and_em_first(self):
        # A pooled majority would pick .1, while equal-domain EM picks .2.
        ds = ['nq'] * 3 + ['triviaqa']
        scores = {0.: np.zeros((4, 2)), .1: np.array([[1, 1]] * 3 + [[0, 0]]),
                  .2: np.array([[1, 1], [0, 0], [0, 0], [1, 1]])}
        self.assertEqual(contract.select_global(scores, ds), .2)
        tied = {.2: scores[.2], .3: scores[.2]}
        self.assertEqual(contract.select_global(tied, ds), .2)

    def test_population_validation_rejects_overlap_and_missing_grid(self):
        dev = [{'stable_id': 'd', 'question': '  WHO\u00a0IS X? ', 'dataset': 'nq'}]
        test = [{'stable_id': 't', 'question': 'who is x?', 'dataset': 'nq'}]
        with self.assertRaises(ValueError): contract.check_disjoint(dev, test)
        with self.assertRaises(ValueError): contract.validate_keys([('a', 'base', 0.)], ['a'])

    def test_empty_condition_is_undefined_not_zero(self):
        self.assertIsNone(contract.transitions([0, 0], [1, 0])['harm_rate'])
        with self.assertRaises(ValueError): contract.transitions([0, 1], [1])

    def test_overlap_normalization_matches_gate_article_punctuation_rules(self):
        self.assertEqual(contract.strong_question_key(' \uff34he,  A! Museum? '), 'museum')


if __name__ == '__main__': unittest.main()
