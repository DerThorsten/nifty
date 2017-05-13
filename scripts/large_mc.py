import sys
sys.path.append("/home/tbeier/bld/nifty/python")

import nifty
import h5py


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
        objective = nifty.graph.multicut.multicutObjective(g, w)


import nifty.graph
import numpy
import random

numpy.random.seed(7)


    


def optimize(objective):

  

   

    if True:


        MincutObjective   = nifty.graph.UndirectedGraph.MincutObjective
        MulticutObjective = nifty.graph.UndirectedGraph.MulticutObjective






        
        # greedy
        greedyFactory = MulticutObjective.greedyAdditiveFactory()
        mincutFactory = MincutObjective.mincutQpboFactory(improve=True)
        #mincutFactory = MincutObjective.greedyAdditiveFactory(improve=True, nodeNumStopCond=0.5)
        multicutFactory = MulticutObjective.multicutIlpFactory(ilpSolver='cplex')
        cgcFactory    = MulticutObjective.cgcFactory(
            doCutPhase=True,doBetterCutPhase=True,
            multicutFactory=multicutFactory,
            doGlueAndCutPhase=True, mincutFactory=mincutFactory,
            nodeNumStopCond=10, sizeRegularizer=1.9)


        


        # greedy+cgc
        chainedSolverFactory = MulticutObjective.chainedSolversFactory(
            multicutFactories=[greedyFactory, cgcFactory]
        )

        solver = cgcFactory.create(objective)

        #solver = MulticutObjective.multicutDecomposer(
        #    submodelFactory=mcFactory,
        #    fallthroughFactory=mcFactory,
        #).create(objective)

        visitor = objective.verboseVisitor(10000)
        with nifty.Timer("fo"):
            arg = solver.optimize(visitor)
            print("cgc res",objective.evalNodeLabels(arg))





      
   




optimize(objective)