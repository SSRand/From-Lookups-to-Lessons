import unittest
import numpy as np
from qa.baselines.bm25_rag.corpus import align_batches, iter_chunks


class CorpusTests(unittest.TestCase):
    def test_removed_suffix_and_process_boundary(self):
        # Three raw batches: 0:3, 3:5 (worker boundary), 5:7.
        rows = [[1,10,11], [1,20,21], [1,30,31], [1,40,41], [1,50], [1,60,61], [1,70,71]]
        batches = [sum(rows[:3], [])[:8], sum(rows[3:5], [])[:4], sum(rows[5:], [])[:4]]
        keep = align_batches(rows, batches, block_size=4)
        self.assertEqual(keep, [3,3,2,3,1,3,1])

    def test_changed_training_token_rejected(self):
        with self.assertRaisesRegex(ValueError, 'token'):
            align_batches([[1,10,11],[1,20,21]], [[1,10,99,1]], block_size=4)

    def test_native_rows_merge_but_heldout_and_title_break(self):
        records = [dict(id='0',title='A',text='a b c',keep=True),
                   dict(id='1',title='A',text='d e',keep=True),
                   dict(id='2',title='A',text='HIDDEN',keep=False),
                   dict(id='3',title='A',text='f g',keep=True),
                   dict(id='4',title='B',text='h i',keep=True)]
        chunks = list(iter_chunks(records, words=4))
        self.assertEqual([r['passage'] for r in chunks], ['a b c d','e','f g','h i'])
        self.assertEqual([r['title'] for r in chunks], ['A','A','A','B'])
        self.assertTrue(all(r['text']==r['title']+'\n'+r['passage'] for r in chunks))

    def test_truncated_prefix_forces_gap(self):
        records=[dict(id='0',title='A',text='a b',keep=True,gap_after=True),
                 dict(id='1',title='A',text='c d',keep=True)]
        self.assertEqual([r['passage'] for r in iter_chunks(records,words=4)],['a b','c d'])

    def test_duplicate_or_unordered_ids_rejected(self):
        rows=[dict(id='0',title='A',text='a',keep=True)]*2
        with self.assertRaisesRegex(ValueError,'order'):
            list(iter_chunks(rows))

if __name__=='__main__': unittest.main()
