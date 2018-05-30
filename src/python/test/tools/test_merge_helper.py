import unittest

import numpy as np
import nifty.tools as nt


class TestMergeHelper(unittest.TestCase):
    uv_ids = np.array([[0, 1],
                       [0, 2],
                       [0, 3],
                       [0, 1],
                       [0, 2],
                       [0, 3],
                       [1, 2]], dtype='uint64')
    indicator = np.array([1,
                          0,
                          1,
                          1,
                          1,
                          0,
                          1], dtype='uint64')

    def test_compute_votes(self):
        sizes = np.zeros_like(self.indicator, dtype='uint64')
        uv_ids_merged, indicators_merged = nt.computeMergeVotes(self.uv_ids, self.indicator, sizes, False)
        unique_ids = np.unique(self.uv_ids, axis=0)
        self.assertEqual(len(uv_ids_merged), len(unique_ids))
        self.assertTrue((uv_ids_merged == unique_ids).all())
        self.assertEqual(len(indicators_merged), len(unique_ids))

        expected_indicator_nom = np.array([2, 1, 1, 1])
        expected_indicator_den = np.array([2, 2, 2, 1])

        self.assertTrue((indicators_merged[:, 0] == expected_indicator_nom).all())
        self.assertTrue((indicators_merged[:, 1] == expected_indicator_den).all())

    def test_merge_votes(self):
        sizes = np.zeros_like(self.indicator, dtype='uint64')
        uv_ids_merged_a, indicators_merged_a = nt.computeMergeVotes(self.uv_ids, self.indicator, sizes, False)
        uv_ids_merged_b, indicators_merged_b = nt.computeMergeVotes(self.uv_ids, self.indicator, sizes, False)

        uv_ids_merged, indicators_merged = nt.mergeMergeVotes(np.concatenate([uv_ids_merged_a,
                                                                              uv_ids_merged_b], axis=0),
                                                             np.concatenate([indicators_merged_a,
                                                                             indicators_merged_b], axis=0))

        unique_ids = np.unique(self.uv_ids, axis=0)
        self.assertEqual(len(uv_ids_merged), len(unique_ids))
        self.assertTrue((uv_ids_merged == unique_ids).all())
        self.assertEqual(len(indicators_merged), len(unique_ids))

        expected_indicator_nom = 2 * np.array([2, 1, 1, 1])
        expected_indicator_den = 2 * np.array([2, 2, 2, 1])

        self.assertTrue((indicators_merged[:, 0] == expected_indicator_nom).all())
        self.assertTrue((indicators_merged[:, 1] == expected_indicator_den).all())



if __name__ == '__main__':
    unittest.main()
