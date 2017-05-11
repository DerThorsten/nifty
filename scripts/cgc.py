import sys
sys.path.append("/home/tbeier/bld/nifty/python")
import nifty




import nifty.graph
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
        obj = nifty.graph.multicut.multicutObjective(g,w)
        
        return obj


    def testCgc(self):

        SubObj =  nifty.graph.multicut.MulticutObjectiveUndirectedGraph


        objective = self.gridModel(gridSize=[15,15])
        submodelFactory = SubObj.greedyAdditiveFactory()
        submodelFactory = SubObj.cgcFactory()

        # we start with the multicut decomposer
        #solver = objective.multicutDecomposer(submodelFactory=submodelFactory).create(objective)
        #visitor = objective.verboseVisitor(1)
        #start  = None
        #arg = solver.optimize(visitor)



        # solver = objective.greedyAdditiveFactory().create(objective)
        # #visitor = objective.empty(600)
        # #start  = None
        # arg = solver.optimize()
        
        MincutObjective = nifty.graph.UndirectedGraph.MincutObjective

        #mincutFactory= MincutObjective.mincutQpboFactory()




  
        mincutFactory= MincutObjective.greedyAdditiveFactory(nodeNumStopCond=0.9, improve=True)

        solver = objective.cgcFactory(mincutFactory=mincutFactory).create(objective)
        visitor = objective.verboseVisitor(1)
        start  = None
        arg = solver.optimize(visitor)






t = TestLiftedMulticutSolver()
t.testCgc()