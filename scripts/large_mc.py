import sys
sys.path.append("/home/tbeier/bld/nifty/python")

import nifty
import h5py


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

    if False:
        # we start with the multicut decomposer
        solver = objective.multicutDecomposer().create(objective)
        visitor = objective.verboseVisitor(1)
        start  = None
        arg = solver.optimize(visitor)



    # solver = objective.greedyAdditiveFactory().create(objective)
    # #visitor = objective.empty(600)
    # #start  = None
    # arg = solver.optimize()

   

    if True:


        MincutObjective   = nifty.graph.UndirectedGraph.MincutObjective
        MulticutObjective = nifty.graph.UndirectedGraph.MulticutObjective






        
        # greedy
        greedyFactory = MulticutObjective.greedyAdditiveFactory()
        mincutQpboFactory    = MincutObjective.mincutQpboFactory(improve=False)

        # cgc
        #mincutFactory = MincutObjective.greedyAdditiveFactory(improve=False,nodeNumStopCond=0.1)
        cgcFactory    = MulticutObjective.cgcFactory(doCutPhase=False, mincutFactory=mincutQpboFactory)


        


        # greedy+cgc
        chainedSolverFactory = MulticutObjective.chainedSolversFactory(
            multicutFactories=[greedyFactory, cgcFactory]
        )

        solver = chainedSolverFactory.create(objective)

        #solver = MulticutObjective.multicutDecomposer(
        #    submodelFactory=mcFactory,
        #    fallthroughFactory=mcFactory,
        #).create(objective)

        visitor = objective.verboseVisitor(500)
        #start  = None
        arg = solver.optimize(visitor)






      
   




optimize(objective)