from __future__ import print_function

import unittest
import numpy

import nifty
import nifty.graph
import nifty.graph.agglo
nagglo = nifty.graph.agglo


class TestAgglo(unittest.TestCase):
    def testUndirectedGraph(self):
        g = nifty.graph.UndirectedGraph(4)
        edges = numpy.array([[0,1], [0,2], [0,3]], dtype='uint64')
        g.insertEdges(edges)

        edgeIndicators = numpy.ones(shape=[g.numberOfEdges])
        edgeSizes = numpy.ones(shape=[g.numberOfEdges])
        nodeSizes = numpy.ones(shape=[g.numberOfNodes])

        clusterPolicy = nagglo.edgeWeightedClusterPolicy(
            graph=g, edgeIndicators=edgeIndicators,
            edgeSizes=edgeSizes, nodeSizes=nodeSizes)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        agglomerativeClustering.run()

        # TODO actually test something
        seg = agglomerativeClustering.result()


if __name__ == '__main__':
    unittest.main()
