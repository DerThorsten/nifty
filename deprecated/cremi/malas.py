import nifty
import nifty.graph
import nifty.graph.agglo

import numpy



G = nifty.graph.UndirectedGraph
nagglo = nifty.graph.agglo


def testMala():



    s = 1000
    g =  nifty.graph.UndirectedGraph(s*s)

    def node(x,y):
        return y+s*x

    for x in range(s):
        for y in range(s):
            if x+1 < s:
                g.insertEdge(node(x,y),node(x+1,y))
            if y+1 < s:
                g.insertEdge(node(x,y),node(x,y+1))  


    
    edgeIndicators = numpy.random.rand(g.numberOfEdges)
    edgeSizes = numpy.ones(shape=[g.numberOfEdges])
    nodeSizes = numpy.ones(shape=[g.numberOfNodes])

    clusterPolicy = nagglo.malaClusterPolicy(
        graph=g, edgeIndicators=edgeIndicators,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes,verbose=True)


    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(verbose=True)

    seg = agglomerativeClustering.result()#out=[1,2,3,4])



testMala()
