import unittest
import random
import numpy

import nifty
import nifty.graph
import nifty.graph.opt.mincut


class TestMincutSolver(unittest.TestCase):
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

    def gridModel(self, gridSize = [5,5],  weightRange = [-2,1]):
        g,nid = self.generateGrid(gridSize)
        d = weightRange[1] - weightRange[0]
        w = numpy.random.rand(g.numberOfEdges)*d + float(weightRange[0])
        print(w.min(),w.max())
        obj = nifty.graph.opt.mincut.mincutObjective(g,w)
        return obj

    @unittest.skipUnless(nifty.Configuration.WITH_QPBO, "need qpbo")
    def testMincutQpbo(self):
        objective = self.gridModel(gridSize=[4,4])
        solver = objective.mincutQpboFactory(improve=False).create(objective)
        visitor = objective.verboseVisitor(1)
        arg = solver.optimize(visitor)

    @unittest.skipUnless(nifty.Configuration.WITH_QPBO, "need qpbo")
    def testMincutQpboImprove(self):
        objective = self.gridModel(gridSize=[8,8])
        solver = objective.mincutQpboFactory(improve=True).create(objective)
        visitor = objective.verboseVisitor(1)
        arg = numpy.zeros(4,dtype='uint8')
        arg = solver.optimize(visitor)


if __name__ == '__main__':
    unittest.main()
