


import unittest
import nifty
import nifty.graph
import numpy
import random

numpy.random.seed(7)

class TestLiftedMulticutSolver(unittest.TestCase):

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
        obj = nifty.graph.multicut.multicutObjective(g,w)
        
        return obj


    def _testGridModelImpl(self, factory, gridSize=[6,6]):

        objective = self.gridModel(gridSize=gridSize)

        # with verbose visitor
        solver = factory.create(objective)
        visitor = objective.verboseVisitor(1000)
        arg = solver.optimize(visitor)

        # without any visitor
        solver = factory.create(objective)
        arg = solver.optimize()



    if nifty.Configuration.WITH_QPBO:
        def testCgc(self):
            objective = self.gridModel(gridSize=[6,6])
            solver = objective.cgcFactory(True,True).create(objective)
            visitor = objective.verboseVisitor(3)
            arg = solver.optimize(visitor)


    def testGreedyAdditive(self):

        Obj = nifty.graph.UndirectedGraph.MulticutObjective
        self._testGridModelImpl(Obj.greedyAdditiveFactory(), gridSize=[6,6])






