from __future__ import print_function


import unittest
import numpy

import nifty
import nifty.graph
import nifty.graph.agglo
nagglo = nifty.graph.agglo


class TestGASP(unittest.TestCase):
    def setUp(self):
        # Create a small graph:
        self.g = g = nifty.graph.UndirectedGraph(4)
        edges = numpy.array([
            [0, 1], [0, 2], [0, 3],
            [1, 3],
            [2, 3]
        ], dtype='uint64')
        g.insertEdges(edges)

        self.edgeIndicators = numpy.array([-10, -2, 6, 3, 11], dtype='float32')

    def test_gasp_average(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='mean',
            add_cannot_link_constraints=False)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result()
        self.assertTrue(seg[0] != seg[1] and seg[1] == seg[2] and seg[2] == seg[3])

    def test_gasp_abs_max(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='abs_max')

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result()
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])


    def test_gasp_sum(self):
        clusterPolicy = nagglo.get_GASP_policy(
            graph=self.g,
            signed_edge_weights=self.edgeIndicators,
            linkage_criteria='sum',
            add_cannot_link_constraints=False)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()
        seg = agglomerativeClustering.result().tolist()
        self.assertTrue(seg[0] != seg[1] and seg[0] == seg[2] and seg[0] == seg[3])



if __name__ == '__main__':
    unittest.main()