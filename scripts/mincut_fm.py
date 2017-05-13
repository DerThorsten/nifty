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
        
        return obj,w


    def testCgc(self):

     

        obj,weights = self.gridModel(gridSize=[4,3])
      

        for edge in obj.graph.edges():
            u = obj.graph.u(edge)
            v = obj.graph.v(edge)

            print(u,"--",v,"  ",weights[edge])

        

        solver = obj.greedyAdditiveFactory(nodeNumStopCond=0.999,improve=False).create(obj)
        visitor = obj.verboseVisitor()
        arg = solver.optimize(visitor)
        print("b",obj.evalNodeLabels(arg))



    

        

        solver = obj.mincutQpboFactory(False).create(obj)
        visitor = obj.verboseVisitor(100)
        start  = None
        arg = solver.optimize(visitor)
        print("b",obj.evalNodeLabels(arg))





t = TestLiftedMulticutSolver()
t.testCgc()