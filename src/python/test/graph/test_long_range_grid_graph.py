from __future__ import print_function
import unittest


import numpy as np
import nifty.graph as ngraph


class TestUndirectedLongRangeGridGraph(unittest.TestCase):
    shape_2d = 2 * (3,)
    offsets_2d = [
        [0, 2],
        [1, 0],
    ]
    offsets_probabilities_2d = [0.3, 0.9]

    def test_undirected_long_range_grid_graph_2d(self):
        g = ngraph.undirectedLongRangeGridGraph(list(self.shape_2d),
                                                self.offsets_2d)
        self.assertTrue(g.numberOfNodes == 9)
        self.assertTrue(g.numberOfEdges == 9)

        # Call some extra methods:
        edges_ID = g.projectEdgesIDToPixels()
        nodes_ID = g.projectNodesIDToPixels()

        edge_values = g.edgeValues(np.random.random(self.shape_2d + (len(self.offsets_2d),)).astype('float32'))
        node_values = g.nodeValues(np.random.random(self.shape_2d).astype('float32'))


    def test_edge_mask_2d(self):
        np.random.seed(42)
        edge_mask = np.random.random((len(self.offsets_2d),) + self.shape_2d) > 0.5
        g1 = ngraph.undirectedLongRangeGridGraph(list(self.shape_2d),
                                                 self.offsets_2d,
                                                 offsets_probabilities=self.offsets_probabilities_2d,
                                                 )
        g2 = ngraph.undirectedLongRangeGridGraph(list(self.shape_2d),
                                                 self.offsets_2d,
                                                 edge_mask=edge_mask,
                                                 )
        # self.assertEqual(g2.numberOfEdges, 4)


if __name__ == '__main__':
    unittest.main()
