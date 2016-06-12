from __future__ import print_function
import nifty
import numpy

def test_undirected_graph():

    g =  nifty.graph.UndirectedGraph(4)
    edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='uint64')
    g.insertEdges(edges)

    edgeList = [e for e in g.edges()]
    assert edgeList == [0,1,2]

    nodeList = [e for e in g.nodes()]
    assert nodeList == [0,1,2,3]

    assert g.u(0) == 0 
    assert g.v(0) == 1 
    assert g.u(1) == 0 
    assert g.v(1) == 2 
    assert g.u(2) == 0 
    assert g.v(2) == 3 

test_undirected_graph()




def test_multicut():

    # build the graph
    shape = 2,2
    g =  nifty.graph.UndirectedGraph(shape[0]*shape[1])
    def f(x,y):
        return x + shape[0]*y

    for y in range(shape[0]):
        for x in range(shape[1]):
            u = f(x, y)
            if x + 1 < shape[0]:
                v = f(x + 1, y)
                g.insertEdge(u, v)
            if y + 1 < shape[1]:
                v = f(x, y + 1)
                g.insertEdge(u, v)

    w = numpy.array(
        [4,5,6,7]
    )
    obj = nifty.graph.multicut.multicutObjective(g,w)
    setttings = nifty.graph.multicut.MulticutGreedyAdditiveSettingsUndirectedGraph()
    factory = nifty.graph.multicut.MulticutGreedyAdditiveFactoryUndirectedGraph(setttings)


    solver = factory.create(obj)
    ret = solver.optimize()

    print(ret) 


    #print(factory)
    
test_multicut()

import sys
print(sys.version)
