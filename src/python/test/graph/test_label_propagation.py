import unittest

from nifty.graph import run_label_propagation, UndirectedGraph
import numpy as np


class TestLabelPropagation(unittest.TestCase):

    def test_label_prop_one_iter(self):
        # Test on a simple star-shaped graph:
        graph = UndirectedGraph(5)
        uv_ids = np.array([
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ])

        # Use some signed edge weights:
        edge_weights = np.array([
            1.,
            1.,
            -1.,
            -1.,
        ])
        graph.insertEdges(uv_ids)

        node_labels = run_label_propagation(graph, edge_values=edge_weights, nb_iter=1, local_edges=None,
                              size_constr=-1,
                              nb_threads=1)

        self.assertIn(node_labels[0], [node_labels[1], node_labels[2]])

    def test_label_prop_connected_components(self):
        # Test on a simple two-components graph:
        graph = UndirectedGraph(5)
        uv_ids = np.array([
            [0, 1],
            [0, 2],
            [1, 2],
            [3, 4],
        ])
        graph.insertEdges(uv_ids)

        node_labels = run_label_propagation(graph, nb_iter=30)

        self.assertNotEqual(node_labels[0],node_labels[3])
        self.assertEqual(node_labels[0],node_labels[1])
        self.assertEqual(node_labels[0],node_labels[2])
        self.assertEqual(node_labels[3],node_labels[4])


if __name__ == '__main__':
    unittest.main()
