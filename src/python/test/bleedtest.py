import opengm
import vigra
import nifty
import numpy

for x in range(20):

    # build the graph
    shape = 40,40
    nNodes = shape[0] * shape[1]
    g =  nifty.graph.UndirectedGraph(shape[0]*shape[1])
    def f(x,y):
        return x + shape[0]*y

    uvs = []
    for y in range(shape[0]):
        for x in range(shape[1]):
            u = f(x, y)
            if x + 1 < shape[0]:
                v = f(x + 1, y)
                g.insertEdge(u, v)
                uvs.append((u,v))
            if y + 1 < shape[1]:
                v = f(x, y + 1)
                g.insertEdge(u, v)
                uvs.append((u,v))
    weights = numpy.random.rand(g.numberOfEdges)-0.5
    weights[numpy.abs(weights)<0.00001] = 0.0001
    uvs = numpy.array(uvs)



    nFac = weights.shape[0]

    gm2 = opengm.gm(numpy.ones(nNodes)*nNodes)
    pf = opengm.pottsFunctions([nNodes,nNodes],numpy.zeros(nFac),weights)
    fid = gm2.addFunctions(pf)
    gm2.addFactors(fid,uvs)


    g =  nifty.graph.UndirectedGraph(int(nNodes))
    g.insertEdges(uvs)
    assert g.numberOfEdges == weights.shape[0]
    assert g.numberOfEdges == uvs.shape[0]
    obj = nifty.graph.multicut.multicutObjective(g, weights)


    with vigra.Timer("ilp-cplex"):
        solver = nifty.multicutIlpFactory(ilpSolver='cplex',verbose=1,
            addThreeCyclesConstraints=False
        ).create(obj)
        ret = solver.optimize()
    print("ilp-cplex",obj.evalNodeLabels(ret))


    with vigra.Timer("opengm"):
        param = opengm.InfParam(workflow="(IC)(CC-IFD)")
        inf = opengm.inference.Multicut(gm2,parameter=param)
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





    fails = 0
    for uv in uvs:
        eshould = arg[uv[0]] != arg[uv[1]]
        enifty  = ret[uv[0]] != ret[uv[1]]
        if enifty != eshould:
            fails +=1
            print uv,eshould,enifty
    assert fails == 0
    print "nFails ",fails
