import opengm
import vigra
import nifty
import numpy


f = "/home/tbeier/Desktop/mc_models/knott-3d-150/gm_knott_3d_039.h5"
f = "/home/tbeier/Desktop/mc_models/knot-3d-550/gm_knott_3d_119.h5"
f = "/home/tbeier/Desktop/mc_models/knott-3d-450/gm_knott_3d_103.h5"
#f = "/home/tbeier/Downloads/gm_large_3.gm"
f = "/home/tbeier/Downloads/gm_small_1.gm"
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


print weights.min(),weights.max()


g =  nifty.graph.UndirectedGraph(int(nNodes))
g.insertEdges(uvs)
assert g.numberOfEdges == weights.shape[0]
assert g.numberOfEdges == uvs.shape[0]
obj = nifty.graph.multicut.multicutObjective(g, weights)


if True:

    greedy=nifty.greedyAdditiveFactory().create(obj)
    ret = greedy.optimize()
    print("greedy",obj.evalNodeLabels(ret))
    with vigra.Timer("fm"):
        ilpFac = nifty.multicutIlpFactory(ilpSolver='cplex',verbose=0,
            addThreeCyclesConstraints=True,
            addOnlyViolatedThreeCyclesConstraints=True
        )
        greedy=nifty.greedyAdditiveFactory()
        factory = nifty.fusionMoveBasedFactory(
            verbose=1,
            #fusionMove=nifty.fusionMoveSettings(mcFactory=greedy),
            fusionMove=nifty.fusionMoveSettings(mcFactory=ilpFac),
            #proposalGen=nifty.greedyAdditiveProposals(sigma=30,nodeNumStopCond=-1,weightStopCond=0.0),
            proposalGen=nifty.watershedProposals(sigma=1,seedFraction=0.01),
            numberOfIterations=1000,
            numberOfParallelProposals=40,
            stopIfNoImprovement=200,
            fuseN=2,
        )
        solver = factory.create(obj)
        ret = solver.optimize(ret)
    print("fm",obj.evalNodeLabels(ret))


with vigra.Timer("ilp-cplex"):
    solver = nifty.multicutIlpFactory(ilpSolver='cplex',verbose=1,
        addThreeCyclesConstraints=False,
        addOnlyViolatedThreeCyclesConstraints=False
    ).create(obj)
    ret = solver.optimize()
print("ilp-cplex",obj.evalNodeLabels(ret))


# with vigra.Timer("ilp-cplex"):
#     solver = nifty.multicutIlpFactory(ilpSolver='cplex',verbose=1,
#         addThreeCyclesConstraints=False
#     ).create(obj)
#     ret = solver.optimize()
# print("ilp-cplex",obj.evalNodeLabels(ret))


with vigra.Timer("opengm"):
    param = opengm.InfParam(workflow="(IC)(CC-IFD)",initializeWith3Cycles=False,numThreads=1)
    inf = opengm.inference.Multicut(gm,parameter=param)
    inf.infer()
    arg = inf.arg()
print("ogm",obj.evalNodeLabels(arg))







# with vigra.Timer("gac"):
#     solver = nifty.greedyAdditiveFactory(verbose=0).create(obj)
#     ret = solver.optimize()
# print("greedy",obj.evalNodeLabels(ret))


# with vigra.Timer("ilp-gurobi"):
#     solver = nifty.multicutIlpFactory(ilpSolver='gurobi',verbose=1).create(obj)
#     ret = solver.optimize()
# print("ilp-gurobi",obj.evalNodeLabels(ret))





fails = 0
for uv in uvs:
    eshould = arg[uv[0]] != arg[uv[1]]
    enifty  = ret[uv[0]] != ret[uv[1]]
    if enifty != eshould:
        fails +=1
        print uv,eshould,enifty

print "nFails ",fails
