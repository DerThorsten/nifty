


import unittest
import nifty
import nifty.graph
import nifty.graph.opt.minstcut
import numpy
import random

numpy.random.seed(7)

class TestMinstcutObjective(unittest.TestCase):

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



    def testGridModel(self, gridSize = [5,5],  weightRange = [0,1]):

        g,nid = self.generateGrid(gridSize)
        d = weightRange[1] - weightRange[0]
        w = numpy.random.rand(g.numberOfEdges)*d + float(weightRange[0])
        u = numpy.zeros((g.numberOfNodes,2))
        print(w.min(),w.max())
        obj = nifty.graph.opt.minstcut.minstcutObjective(g,w,u)
        labels = numpy.zeros(g.nodeIdUpperBound+1,dtype='uint8')
        value = obj.evalNodeLabels(labels)
        self.assertAlmostEqual(value,0.0)
        return obj

