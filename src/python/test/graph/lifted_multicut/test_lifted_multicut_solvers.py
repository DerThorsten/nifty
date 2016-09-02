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

    def gridLiftedModel(self, gridSize = [3,2], bfsRadius=2, weightRange = [-1,1]):

        g,nid = self.generateGrid(gridSize)
        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
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



    def testLiftedMulticutGreedyAdditive(self):

        obj,nid = self.gridLiftedModel(gridSize=[10,10] , bfsRadius=2, weightRange=[-1,1])
        solverFactory = obj.greedyAdditiveFactory()
        solver = solverFactory.create(obj)
        visitor = obj.verboseVisitor()
        arg = solver.optimize(visitor=visitor)
        print('\n',arg.reshape([10,10]))






    def implTestLiftedMulticutIlpBfsGrid(self, ilpSolver, gridSize=[4,4], bfsRadius=2, weightRange=[-2,1]):

        obj,nid = self.gridLiftedModel(gridSize=gridSize , bfsRadius=bfsRadius, weightRange=weightRange)
        solverFactory = obj.liftedMulticutIlpFactory(ilpSolver=ilpSolver)
        solver = solverFactory.create(obj)
        visitor = obj.verboseVisitor()
        arg = solver.optimize(visitor=visitor)
        print('\n',arg.reshape(gridSize))


    def testLiftedMulticutIlpBfsGrid(self):

        if nifty.Configuration.WITH_GLPK:
            for x in range(10):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='glpk', gridSize=[3,3], 
                                                      weightRange=(-2,1))    

        if nifty.Configuration.WITH_CPLEX:
            for x in range(10):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='cplex')

        if nifty.Configuration.WITH_GUROBI:
            for x in range(10):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='gurobi')   


