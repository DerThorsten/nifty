import opengm
import vigra
import nifty
import numpy


f = "/home/tbeier/Desktop/mc_models/knott-3d-150/gm_knott_3d_039.h5"
f = "/home/tbeier/Desktop/mc_models/knot-3d-550/gm_knott_3d_119.h5"
#f = "/home/tbeier/Desktop/mc_models/knott-3d-450/gm_knott_3d_103.h5"
#f = "/home/tbeier/Downloads/gm_large_3.gm"
#f = "/home/tbeier/Downloads/gm_small_1.gm"
#f = "/home/tbeier/Desktop/mc_models/knott-3d-300/gm_knott_3d_072.h5"
gm = opengm.loadGm(f)





def opengmToNumpy(gm):
    nNodes = gm.numberOfVariables
    nEdges = gm.numberOfFactors


    factorSubset=opengm.FactorSubset(gm)
    weights = factorSubset.evaluate([0,1])
    weightsB = factorSubset.evaluate([0,0])
    print "wb", weightsB.sum()
    vis =  factorSubset.variableIndices()
    assert len(numpy.unique(vis.reshape(-1))) == nNodes
    #print vis.shape,weights.shape
    assert vis.max()+1 == nNodes
    return nNodes,weights,vis

nNodes,weights, uvs = opengmToNumpy(gm)
nFac = weights.shape[0]


g =  nifty.graph.UndirectedGraph(int(nNodes))
g.insertEdges(uvs)
obj = nifty.graph.multicut.multicutObjective(g, weights)


ilpFactory = obj.multicutIlpFactory(ilpSolver='cplex',
    addThreeCyclesConstraints=True,
    addOnlyViolatedThreeCyclesConstraints=True
    #memLimit= 0.01
)


greedy=obj.greedyAdditiveFactory()
fmFactory = obj.fusionMoveBasedFactory(
    #fusionMove=nifty.fusionMoveSettings(mcFactory=greedy),
    fusionMove=obj.fusionMoveSettings(mcFactory=ilpFactory),
    #proposalGen=nifty.greedyAdditiveProposals(sigma=30,nodeNumStopCond=-1,weightStopCond=0.0),
    proposalGen=obj.watershedProposals(sigma=1,seedFraction=0.01),
    numberOfIterations=300,
    numberOfParallelProposals=16, # no effect if nThreads equals 0 or 1
    numberOfThreads=0,
    stopIfNoImprovement=10,
    fuseN=2,
)



s = g.perturbAndMapSettings(fmFactory)
pAndMap = g.perturbAndMap(obj, s)
print pAndMap.optimize()


