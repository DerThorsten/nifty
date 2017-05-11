import sys
sys.path.append("/home/tbeier/bld/nifty/python")
import nifty
import h5py



import nifty.graph
#import nifty.graph.optimization
import nifty.graph.optimization.mincut

import numpy
import random

numpy.random.seed(7)

class TestLiftedMulticutSolver():

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



    def gridModel(self, gridSize = [5,5],  weightRange = [-1,1]):

        g,nid = self.generateGrid(gridSize)
        d = weightRange[1] - weightRange[0]
        w = numpy.random.rand(g.numberOfEdges)*d + float(weightRange[0])
        print(w.min(),w.max())
        obj = nifty.graph.mincut.mincutObjective(g,w)
        
        return obj


    def testCgc(self):

        if True:
            file = "/home/tbeier/datasets/large_mc_problems/sampleD_subsample_reduced_model.h5"
            h5File = h5py.File(file,'r')

            with nifty.Timer("load serialization"):
                serialization = h5File['graph'][:]

            with nifty.Timer("deserialize"):
                g = nifty.graph.UndirectedGraph()
                g.deserialize(serialization)

            with nifty.Timer("load costs"):
                w = h5File['costs'][:]

            with nifty.Timer("setup objective"):
                obj = nifty.graph.mincut.mincutObjective(g, w)

        else:

            obj = self.gridModel(gridSize=[100,100])
      


        if False:

            pgen = obj.watershedProposalGenerator(sigma=1.0, numberOfSeeds=0.01)
            solver = obj.mincutCcFusionMoveBasedFactory(
                proposalGenerator=pgen
            ).create(obj)
            visitor = obj.verboseVisitor(1)
            start  = None
            arg = solver.optimize(visitor)

            print("a",obj.evalNodeLabels(arg))




        if True:

        

            solver = obj.greedyAdditiveFactory(improve=False).create(obj)
            visitor = obj.verboseVisitor(100)
            start  = None
            arg = solver.optimize(visitor)
            print("b",obj.evalNodeLabels(arg))



        if False:

        

            solver = obj.mincutQpboFactory(True).create(obj)
            visitor = obj.verboseVisitor(100)
            start  = None
            arg = solver.optimize(visitor)
            print("b",obj.evalNodeLabels(arg))





t = TestLiftedMulticutSolver()
t.testCgc()