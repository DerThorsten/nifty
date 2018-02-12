import unittest

import numpy as np
import nifty.tools as nt


class TestEdgeMapping(unittest.TestCase):

    def test_edge_mapping_toy(self):
        uv_ids = np.array([[0, 1],
                           [0, 2],
                           [1, 2],
                           [2, 3],
                           [2, 4],
                           [2, 5],
                           [2, 6],
                           [3, 4],
                           [3, 6],
                           [4, 5]], dtype='int64')

        # node labeling           0  1  2  3  4  5  6
        node_labeling = np.array([0, 1, 0, 2, 1, 2, 3], dtype='uint64')
        self.assertEqual(len(node_labeling), len(np.unique(uv_ids)))
        edge_mapping = nt.EdgeMapping(uv_ids, node_labeling)

        new_uv_ids = edge_mapping.newUvIds()
        new_uv_ids_exp = np.array([[0, 1],
                                   [0, 2],
                                   [0, 3],
                                   [1, 2],
                                   [2, 3]], dtype='uint64')

        self.assertEqual(new_uv_ids.shape, new_uv_ids_exp.shape)
        self.assertTrue((new_uv_ids == new_uv_ids_exp).all())

        # test edge mappings
        # mapping
        # edge 0: 0 -> 1 == 0 -> 1 : 0
        # edge 1: 0 -> 2 == 0 -> 0 : Null
        # edge 2: 1 -> 2 == 1 -> 0 : 0
        # edge 3: 2 -> 3 == 0 -> 2 : 1
        # edge 4: 2 -> 4 == 0 -> 1 : 0
        # edge 5: 2 -> 5 == 0 -> 2 : 1
        # edge 6: 2 -> 6 == 0 -> 3 : 2
        # edge 7: 3 -> 4 == 2 -> 1 : 3
        # edge 8: 3 -> 6 == 2 -> 3 : 4
        # edge 9: 4 -> 5 == 1 -> 2 : 3

        edge_values = np.ones(len(uv_ids), dtype='float32')
        # sum mapping
        new_values = edge_mapping.mapEdgeValues(edge_values, "sum")
        new_values_exp = np.array([3, 2, 1, 2, 1], dtype='float32')
        self.assertEqual(new_values.shape, new_values_exp.shape)
        self.assertTrue(np.allclose(new_values, new_values_exp))

        # mean mapping
        new_values = edge_mapping.mapEdgeValues(edge_values, "mean")
        new_values_exp = np.array([1, 1, 1, 1, 1], dtype='float32')
        self.assertEqual(new_values.shape, new_values_exp.shape)
        self.assertTrue(np.allclose(new_values, new_values_exp))

        edge_values = np.ones(len(uv_ids), dtype='float32')
        edge_values[::2] = 0
        # min mapping
        # edges:  0 1 2 3 4 5 6 7 8 9
        # values: 0 1 0 1 0 1 0 1 0 1
        new_values = edge_mapping.mapEdgeValues(edge_values, "min")
        new_values_exp = np.array([0, 1, 0, 1, 0], dtype='float32')
        self.assertEqual(new_values.shape, new_values_exp.shape)
        self.assertTrue(np.allclose(new_values, new_values_exp))

        # max mapping
        # edges:  0 1 2 3 4 5 6 7 8 9
        # values: 0 1 0 1 0 1 0 1 0 1
        new_values = edge_mapping.mapEdgeValues(edge_values, "max")
        new_values_exp = np.array([0, 1, 0, 1, 0], dtype='float32')
        self.assertEqual(new_values.shape, new_values_exp.shape)
        self.assertTrue(np.allclose(new_values, new_values_exp))


if __name__ == '__main__':
    unittest.main()
