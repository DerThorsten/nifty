from __future__ import print_function
import nifty
import numpy



G = nifty.graph.UndirectedGraph
CG = G.EdgeContractionGraph

GMCO = G.MulticutObjective
CGMCO = CG.MulticutObjective



def testEdgeContractionGraph():

    g =  nifty.graph.UndirectedGraph(4)
    edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='uint64')
    g.insertEdges(edges)





    class MyCb(nifty.graph.EdgeContractionGraphCallback):
        def __init__(self):
            super(MyCb, self).__init__()

            self.nCallsContractEdge = 0
            self.nCallsMergeEdges = 0
            self.nCallsMergeNodes = 0
            self.nCallscontractEdgeDone = 0

        def contractEdge(self, edge):
            self.nCallsContractEdge += 1

        def mergeEdges(self, alive, dead):
            self.nCallsMergeEdges += 1

        def mergeNodes(self, alive, dead):
            self.nCallsMergeNodes += 1

        def contractEdgeDone(self, edge):
            self.nCallscontractEdgeDone += 1



    cb = MyCb()
    ecg = nifty.graph.edgeContractionGraph(g,cb)
    ecg.contractEdge(0)

    assert cb.nCallsContractEdge == 1
    assert cb.nCallsMergeNodes == 1
    assert cb.nCallscontractEdgeDone == 1

