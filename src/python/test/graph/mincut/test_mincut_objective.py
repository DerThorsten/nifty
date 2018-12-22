import unittest
import numpy
import random

import nifty


class TestMincutObjective(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(7)

    def generateGrid(self, gridSize):
        def nid(x, y):
            return x*gridSize[1] + y
        G = nifty.graph.UndirectedGraph
        g =  G(gridSize[0] * gridSize[1])
        for x in range(gridSize[0]):
            for y in range(gridSize[1]):

                u = nid(x,y)

                if x + 1 < gridSize[0]:

                    v = nid(x+1, y)
                    g.insertEdge(u, v)

                if y + 1 < gridSize[1]:

                    v = nid(x, y+1)
                    g.insertEdge(u, v)

        return g, nid

    def testGridModel(self, gridSize = [5,5],  weightRange = [-2,1]):

        beta = .7
        gridSize = [5, 5]
        weightRange = [0, 1]

        g, nid = self.generateGrid(gridSize)

        w = numpy.zeros(g.numberOfEdges)
        labels = numpy.zeros(g.nodeIdUpperBound+1, dtype='uint8')

        for x in range(gridSize[0]):
            for y in range(gridSize[1]):

                nodeId = nid(x,y)
                if x in range(1,gridSize[0] - 1):
                    if y in range(1,gridSize[1] - 1):
                        labels[nodeId] = 1
                    else:
                        labels[nodeId] = 0
                else:
                    labels[nodeId] = 0

        for x in range(gridSize[0]):
            for y in range(gridSize[1]):

                p = nid(x, y)

                if x + 1 < gridSize[0]:
                    q = nid(x+1,y)
                    weightId = g.findEdge(p,q)
                    if (labels[p] == labels[q]):
                        w[weightId] = 0
                    else:
                        w[weightId] = beta
                if y + 1 < gridSize[1]:
                    q = nid(x, y + 1)
                    weightId = g.findEdge(p,q)
                    if (labels[p] == labels[q]):
                        w[weightId] = 0
                    else:
                        w[weightId] = beta

        obj = nifty.graph.opt.mincut.mincutObjective(g, w)

        value = obj.evalNodeLabels(labels)
        self.assertAlmostEqual(value, 8.4)
        return obj


if __name__ == '__main__':
    unittest.main()
