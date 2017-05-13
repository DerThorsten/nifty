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

        MincutObjective   = nifty.graph.UndirectedGraph.MincutObjective
        MulticutObjective = nifty.graph.UndirectedGraph.MulticutObjective


        SubObj =  nifty.graph.multicut.MulticutObjectiveUndirectedGraph


        objective = self.gridModel(gridSize=[20,20])
        # greedy
        greedyFactory = MulticutObjective.greedyAdditiveFactory()
        mincutFactory = MincutObjective.mincutQpboFactory(improve=False)
        #mincutFactory = MincutObjective.greedyAdditiveFactory(improve=True)

        multicutFactory = MulticutObjective.multicutIlpFactory(ilpSolver='cplex')

        solver    = objective.cgcFactory(
        doCutPhase=True,doBetterCutPhase=True,
        doGlueAndCutPhase=True, 
        mincutFactory=mincutFactory,
        multicutFactory=multicutFactory,
        nodeNumStopCond=100, sizeRegularizer=0.9).create(objective)


        arg = solver.optimize(objective.verboseVisitor(1))#,numpy.zeros(10*10))
        objective.evalNodeLabels(arg)

t = TestLiftedMulticutSolver()
t.testCgc()