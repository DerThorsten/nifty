import opengm
import vigra
import nifty
import numpy

f = "/home/tbeier/Desktop/mc_models/knott-3d-300/gm_knott_3d_079.h5"
#f = "/home/tbeier/Desktop/mc_models/knott-3d-150/gm_knott_3d_032.h5"
gm = opengm.loadGm(f)
weights = []
uvs = []

for f in range(gm.numberOfFactors):
    fac = gm[f]
    v00 = fac[0,0]
    v01 = fac[0,1]
    uv = fac.variableIndices
    weights.append(v01-v00)
    uvs.append(uv)
    #print v00,v01,uv

weights = numpy.array(weights)
uvs = numpy.array(uvs)

nNodes = uvs.max()+1

g =  nifty.graph.UndirectedGraph(int(nNodes))
g.insertEdges(uvs)
obj = nifty.graph.multicut.multicutObjective(g, weights)



with vigra.Timer("nifty fm"):

    factory = nifty.graph.multicut.MulticutGreedyAdditiveFactoryUndirectedGraph()
    fmSettings = nifty.graph.multicut.FusionMoveSettingsUndirectedGraph()
    fmSettings.mcFactory = factory


    setttings = nifty.graph.multicut.FusionMoveBasedGreedyAdditiveSettingsUndirectedGraph()
    setttings.numberOfIterations = 10
    setttings.fusionMoveSettings = fmSettings
    #setttings.addOnlyViolatedThreeCyclesConstraints = True
    fmFactory = nifty.graph.multicut.FusionMoveBasedGreedyAdditiveFactoryUndirectedGraph(setttings)
    fmSolver = fmFactory.create(obj)
    ret = fmSolver.optimize()
print("fm",obj.evalNodeLabels(ret))

with vigra.Timer("nifty"):
    setttings = nifty.graph.multicut.MulticutIlpCplexSettingsUndirectedGraph()
    setttings.addThreeCyclesConstraints = True
    setttings.addOnlyViolatedThreeCyclesConstraints = True
    mcIlpFactory = nifty.graph.multicut.MulticutIlpCplexFactoryUndirectedGraph(setttings)
    mcIlp = mcIlpFactory.create(obj)
    ret = mcIlp.optimize()
print("ilpres",obj.evalNodeLabels(ret))

sys.exit()

with vigra.Timer("nifty gadd"):
    setttings = nifty.graph.multicut.MulticutGreedyAdditiveSettingsUndirectedGraph()
    setttings.verbose = 0
    #setttings.addOnlyViolatedThreeCyclesConstraints = True
    mcIlpFactory = nifty.graph.multicut.MulticutGreedyAdditiveFactoryUndirectedGraph(setttings)
    mcIlp = mcIlpFactory.create(obj)
    ret = mcIlp.optimize()





with vigra.Timer("opengm"):
    param = opengm.InfParam(workflow="(IC)(CC-IFD)")
    inf = opengm.inference.Multicut(gm,parameter=param)
    inf.infer()
    arg = inf.arg()




for uv in uvs:
    eshould = arg[uv[0]] != arg[uv[1]]
    enifty  = ret[uv[0]] != ret[uv[1]]
    assert enifty == eshould
    #print eshould,enifty
