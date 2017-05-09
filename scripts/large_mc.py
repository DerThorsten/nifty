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


    # we start with the multicut decomposer
    solver = objective.multicutDecomposer().create(objective)
    visitor = objective.verboseVisitor(1)
    start  = None
    arg = solver.optimize(visitor)



    # solver = objective.greedyAdditiveFactory().create(objective)
    # #visitor = objective.empty(600)
    # #start  = None
    # arg = solver.optimize()

    

    # solver = objective.cgcFactory(False,True).create(objective)
    # visitor = objective.verboseVisitor(1)
    # start  = None
    # arg = solver.optimize(visitor, arg)



optimize(objective)