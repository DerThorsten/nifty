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




def make2x2Rag():

    labels = numpy.zeros(shape=[2,2],dtype='uint32')
    print(labels.shape)

    labels[0,0] = 0 
    labels[1,0] = 1 
    labels[0,1] = 0 
    labels[1,1] = 2 

    g =  nifty.graph.rag.explicitLabelsGridRag2D(labels)

    return g

def test_grid_rag():

    labels = numpy.zeros(shape=[2,2],dtype='uint32')
    print(labels.shape)

    labels[0,0] = 0 
    labels[1,0] = 1 
    labels[0,1] = 0 
    labels[1,1] = 2 

    g =  nifty.graph.rag.explicitLabelsGridRag2D(labels)

    insertWorked = True
    try:
        g.insertEdge(0,1)
    except:
        insertWorked = False
    assert insertWorked == False


test_grid_rag()