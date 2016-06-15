import opengm
import vigra
import nifty
import numpy

f = "/home/tbeier/Desktop/mc_models/knott-3d-300/gm_knott_3d_079.h5"
# f = "/home/tbeier/Desktop/mc_models/knott-3d-150/gm_knott_3d_032.h5"
#f = "/home/tbeier/Desktop/mc_models/knot-3d-550/gm_knott_3d_119.h5"
# f = "/home/tbeier/Desktop/mc_models/knott-3d-450/gm_knott_3d_096.h5"
#f = "/home/tbeier/Downloads/gm_large_3.gm"
f = "/home/tbeier/Downloads/gm_small_1.gm"

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
#sys.exit(0)


g =  nifty.graph.UndirectedGraph(int(nNodes))
g.insertEdges(uvs)
obj = nifty.graph.multicut.multicutObjective(g, weights)



with vigra.Timer("ilp-cplex"):
    solver = nifty.multicutIlpFactory(ilpSolver='cplex',verbose=1,
        addThreeCyclesConstraints=False
    ).create(obj)
    ret = solver.optimize()
print("ilp-cplex",obj.evalNodeLabels(ret))

with vigra.Timer("opengm"):
    param = opengm.InfParam(workflow="(IC)(CC-IFD)")
    inf = opengm.inference.Multicut(gm,parameter=param)
    inf.infer()
    arg = inf.arg()
print("ogm",obj.evalNodeLabels(arg))




# with vigra.Timer("fm"):
#     factory = nifty.fusionMoveBasedFactory(
#         verbose=0,
#         fusionMove=nifty.fusionMoveSettings(mcFactory=nifty.greedyAdditiveFactory()),
#         proposalGen=nifty.greedyAdditiveProposals(sigma=2.0,nodeNumStopCond=-1.0)
#     )
#     solver = factory.create(obj)
#     ret = solver.optimize()
# print("fm",obj.evalNodeLabels(ret))


# with vigra.Timer("gac"):
#     solver = nifty.greedyAdditiveFactory(verbose=0).create(obj)
#     ret = solver.optimize()
# print("greedy",obj.evalNodeLabels(ret))


# with vigra.Timer("ilp-gurobi"):
#     solver = nifty.multicutIlpFactory(ilpSolver='gurobi',verbose=1).create(obj)
#     ret = solver.optimize()
# print("ilp-gurobi",obj.evalNodeLabels(ret))






for uv in uvs:
    eshould = arg[uv[0]] != arg[uv[1]]
    enifty  = ret[uv[0]] != ret[uv[1]]
    assert enifty == eshould
    #print eshould,enifty
