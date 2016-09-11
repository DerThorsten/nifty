from __future__ import print_function
import unittest

import nifty
import nifty.graph
nlmc = nifty.graph.lifted_multicut
import numpy
import random



class TestLiftedGraphFeatures(unittest.TestCase):

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

    def gridLiftedModel(self, gridSize = [3,2], bfsRadius=2, weightRange = [-1,1]):

        g,nid = self.generateGrid(gridSize)
        obj = nlmc.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph

    
        # this should  add edges
        obj.insertLiftedEdgesBfs(bfsRadius)
        postEdges = liftedGraph.numberOfEdges


        for edge in liftedGraph.edges():
            u,v = liftedGraph.uv(edge)
            w = random.uniform(weightRange[0],weightRange[1])
            obj.setCost(u, v, w)

        return obj,nid



     
    def testLiftedUcFeatures(self):
        obj,nid = self.gridLiftedModel(gridSize=[100,100],bfsRadius=3)

        graph = obj.graph 

        edgeIndicators = numpy.random.rand(graph.edgeIdUpperBound + 1) + 2.0
        edgeSizes      = numpy.random.rand(graph.edgeIdUpperBound + 1) + 0.5  
        nodeSizes      = numpy.random.rand(graph.nodeIdUpperBound + 1) + 0.5

        features = nlmc.liftedUcmFeatures(
            objective=obj,
            edgeIndicators=edgeIndicators,
            edgeSizes=edgeSizes,
            nodeSizes=nodeSizes,
            sizeRegularizers=[0.01, 0.02, 0.3, 0.4]
        )

        featuresReg = features[::2,:]
        featuresRaw = features[1::2,:]

        self.assertEqual(featuresReg.shape[1], obj.numberOfLiftedEdges)
        self.assertEqual(featuresReg.shape[0], 4)
        self.assertEqual(featuresRaw.shape[0], 4)


        self.assertTrue(numpy.all(numpy.isfinite(featuresReg)))
        self.assertTrue(numpy.all(numpy.isfinite(featuresRaw)))

        self.assertGreaterEqual(featuresRaw.min(), 2.0)
        self.assertLessEqual(featuresRaw.max(), 3.0)


        self.assertGreaterEqual(featuresReg.min(), 0.0)