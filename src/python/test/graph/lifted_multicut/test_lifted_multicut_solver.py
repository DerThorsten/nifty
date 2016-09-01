from __future__ import print_function
import unittest

import nifty
import numpy
import random



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

    def testInsertLiftedEdgesBfsBig(self):

        g,nid = self.generateGrid([10, 10])
        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        graph = obj.graph
        liftedGraph = obj.liftedGraph

    
        # this should  add edges
        obj.insertLiftedEdgesBfs(4)
        postEdges = liftedGraph.numberOfEdges


        for edge in liftedGraph.edges():
            u,v = liftedGraph.uv(edge)
            w = random.uniform(-1,2)
            obj.setCost(u, v, w)

        solverFactory = obj.greedyAdditiveFactory()
        solver = solverFactory.create(obj)

        visitor = obj.verboseVisitor()

        arg = solver.optimize(visitor=visitor)

        print('\n',arg.reshape([10,10]))