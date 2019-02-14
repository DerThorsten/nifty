from __future__ import print_function
import unittest
import random

import numpy
import nifty
import nifty.graph


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

    def gridLiftedModel(self, gridSize=[3, 2], bfsRadius=2, weightRange=[-1, 1]):
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
        for x in range(5):
            obj,nid = self.gridLiftedModel(gridSize=[40,40] , bfsRadius=4, weightRange=[-1,1])


            solverFactory = obj.liftedMulticutGreedyAdditiveFactory()
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            argN = solver.optimize()

            if False:
                solverFactory = obj.liftedMulticutAndresGreedyAdditiveFactory()
                solver = solverFactory.create(obj)
                visitor = obj.verboseVisitor(100)
                argA = solver.optimize()

                self.assertAlmostEqual(obj.evalNodeLabels(argA), obj.evalNodeLabels(argN))

    def testLiftedMulticutKernighanLinSimple(self):
        G = nifty.graph.UndirectedGraph
        g = G(5);
        g.insertEdge(0, 1) # 0
        g.insertEdge(0, 3) # 1
        g.insertEdge(1, 2) # 2
        g.insertEdge(1, 4) #
        g.insertEdge(3, 4) # 4

        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
        lg = obj.liftedGraph

        obj.setCost(0, 1, 10.0)
        obj.setCost(0, 2, -1.0)
        obj.setCost(0, 3, -1.0)
        obj.setCost(0, 4, -1.0)
        obj.setCost(1, 2, 10.0)
        obj.setCost(1, 3, -1.0)
        obj.setCost(1, 4,  4.0)
        obj.setCost(2, 3, -1.0)
        obj.setCost(2, 4, -1.0)
        obj.setCost(3, 4, 10.0)

        arg = numpy.array([1,2,3,4,5])
        solverFactory = obj.liftedMulticutKernighanLinFactory()
        solver = solverFactory.create(obj)
        visitor = obj.verboseVisitor(100)

        arg2 = solver.optimize(visitor,arg)

        self.assertEqual(arg2[0] != arg2[1],  bool(0))
        self.assertEqual(arg2[0] != arg2[2],  bool(0))
        self.assertEqual(arg2[0] != arg2[3],  bool(1))
        self.assertEqual(arg2[0] != arg2[4],  bool(1))
        self.assertEqual(arg2[1] != arg2[2],  bool(0))
        self.assertEqual(arg2[1] != arg2[3],  bool(1))
        self.assertEqual(arg2[1] != arg2[4],  bool(1))
        self.assertEqual(arg2[2] != arg2[3],  bool(1))
        self.assertEqual(arg2[2] != arg2[4],  bool(1))
        self.assertEqual(arg2[3] != arg2[4],  bool(0))

    def testLiftedMulticutKernighanLin(self):
        gridSize = [5, 5]
        for x in range(4):
            obj,nid = self.gridLiftedModel(gridSize=gridSize , bfsRadius=2, weightRange=[-3,1])

            solverFactory = obj.liftedMulticutGreedyAdditiveFactory()
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            arg = solver.optimize()
            argGC = arg.copy()
            egc = obj.evalNodeLabels(argGC)

            #arg = numpy.arange(gridSize[0]*gridSize[1])
            solverFactory = obj.liftedMulticutKernighanLinFactory()
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            arg2 = solver.optimize(argGC.copy())
            ekl = obj.evalNodeLabels(arg2)

            if False:
                solverFactory = obj.liftedMulticutAndresKernighanLinFactory()
                solver = solverFactory.create(obj)
                visitor = obj.verboseVisitor(100)
                arg3 = solver.optimize(argGC.copy())
                eakl = obj.evalNodeLabels(arg3)

                self.assertLessEqual(ekl, egc)
                self.assertAlmostEqual(ekl, eakl)

        for x in range(4):
            obj,nid = self.gridLiftedModel(gridSize=gridSize , bfsRadius=2, weightRange=[-3,1])

            arg = numpy.arange(gridSize[0]*gridSize[1])

            solverFactory = obj.liftedMulticutKernighanLinFactory()
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            arg2 = solver.optimize(arg.copy())
            ekl = obj.evalNodeLabels(arg2)

            if False:
                solverFactory = obj.liftedMulticutAndresKernighanLinFactory()
                solver = solverFactory.create(obj)
                visitor = obj.verboseVisitor(100)
                arg3 = solver.optimize(arg.copy())
                eakl = obj.evalNodeLabels(arg3)

                self.assertAlmostEqual(ekl, eakl)

    def testLiftedMulticutSolverFm(self):
        random.seed(0)
        for x in range(1):
            obj, nid = self.gridLiftedModel(gridSize=[40,40] , bfsRadius=4, weightRange=[-1,1])

            pgen = obj.watershedProposalGenerator(sigma=1.0,
                                                  seedingStrategy='SEED_FROM_LOCAL',
                                                  numberOfSeeds=0.1)
            # print(x)

            solverFactory = obj.fusionMoveBasedFactory()
            solverFactory = obj.fusionMoveBasedFactory(proposalGenerator=pgen,
                                                       numberOfIterations=100,
                                                       stopIfNoImprovement=10)
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            argN = solver.optimize(visitor)

    def implTestLiftedMulticutIlpBfsGrid(self, ilpSolver,
                                         gridSize=[4,4], bfsRadius=4,
                                         weightRange=[-2,1], verbose=0,
                                         relativeGap=0.00001):

        obj,nid = self.gridLiftedModel(gridSize=gridSize,
                                       bfsRadius=bfsRadius,
                                       weightRange=weightRange)
        solverFactory = obj.liftedMulticutIlpFactory(ilpSolver=ilpSolver,
                                                     relativeGap=relativeGap)
        solver = solverFactory.create(obj)
        if verbose > 0:
            visitor = obj.verboseVisitor(100)
            arg = solver.optimize(visitor=visitor)
        else:
            arg = solver.optimize()

    def testLiftedMulticutIlpBfsGrid(self):
        if nifty.Configuration.WITH_GLPK:
            for x in range(5):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='glpk', gridSize=[3,3],
                                                      relativeGap=0.3,
                                                      weightRange=(-2,1))
        if nifty.Configuration.WITH_CPLEX:
            for x in range(5):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='cplex', gridSize=[3,3],
                                                      relativeGap=0.3,
                                                      verbose=0,  weightRange=(-2,1))
        if nifty.Configuration.WITH_GUROBI:
            for x in range(5):
                self.implTestLiftedMulticutIlpBfsGrid(ilpSolver='gurobi')

    def implTestSimpleChainModelOpt(self, size, ilpSolver):
        if nifty.Configuration.WITH_CPLEX:
            #  0 - 1 - 2 - 3
            #
            G = nifty.graph.UndirectedGraph
            g =  G(size)
            for x in range(size - 1):
                g.insertEdge(x, x+1)

            obj = nifty.graph.lifted_multicut.liftedMulticutObjective(g)
            graph = obj.graph
            liftedGraph = obj.liftedGraph

            for x in range(size - 1):
                w = 1.0
                obj.setCost(x, x+1, w)

            w = -1.0 * float(size)
            obj.setCost(0, size-1, w)

            solverFactory = obj.liftedMulticutIlpFactory(ilpSolver=ilpSolver)
            solver = solverFactory.create(obj)
            visitor = obj.verboseVisitor(100)
            arg = solver.optimize()

            self.assertNotEqual(arg[0],arg[size-1])

    def testSimpleChainModelMulticutGlpk(self):
        if False and nifty.Configuration.WITH_GLPK:
            for x in range(10):
                self.implTestSimpleChainModelOpt(ilpSolver='glpk', size=4)

    def testSimpleChainModelMulticutGurobi(self):
        if nifty.Configuration.WITH_GUROBI:
            for x in range(10):
                self.implTestSimpleChainModelOpt(ilpSolver='gurobi', size=4)

    def testSimpleChainModelMulticutCplex(self):
        if nifty.Configuration.WITH_CPLEX:
            for x in range(10):
                self.implTestSimpleChainModelOpt(ilpSolver='cplex', size=4)


if __name__ == '__main__':
    unittest.main()
