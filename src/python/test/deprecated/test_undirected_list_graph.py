from __future__ import print_function
import nifty
import numpy



G = nifty.graph.UndirectedGraph
CG = G.EdgeContractionGraph

GMCO = G.MulticutObjective
CGMCO = CG.MulticutObjective



def testUndirectedGraph():

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


def testUndirectedGraphSerialization():

    gA =  nifty.graph.UndirectedGraph(4)
    edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='uint64')
    gA.insertEdges(edges)

    serialization = gA.serialize()

    g = nifty.graph.UndirectedGraph()

    g.deserialize(serialization)

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





