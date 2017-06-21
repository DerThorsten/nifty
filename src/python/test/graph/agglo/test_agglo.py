from __future__ import print_function
import nifty

import nifty.graph
import nifty.graph.agglo

import numpy



nagglo = nifty.graph.agglo


def testUndirectedGraph():

    g =  nifty.graph.UndirectedGraph(4)
    edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='uint64')
    g.insertEdges(edges)


    edgeIndicators = numpy.ones(shape=[g.numberOfEdges])
    edgeSizes = numpy.ones(shape=[g.numberOfEdges])
    nodeSizes = numpy.ones(shape=[g.numberOfNodes])

    clusterPolicy = nagglo.edgeWeightedClusterPolicy(
        graph=g, edgeIndicators=edgeIndicators,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes)


    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
    agglomerativeClustering.run()

    seg = agglomerativeClustering.result()#out=[1,2,3,4])



def testMala():

    g =  nifty.graph.UndirectedGraph(4)
    edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='uint64')
    g.insertEdges(edges)


    edgeIndicators = numpy.ones(shape=[g.numberOfEdges])
    edgeSizes = numpy.ones(shape=[g.numberOfEdges])
    nodeSizes = numpy.ones(shape=[g.numberOfNodes])

    clusterPolicy = nagglo.malaClusterPolicyWithUcm(
        graph=g, edgeIndicators=edgeIndicators,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes)


    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
    agglomerativeClustering.run()

    # merge graph edge merge times
    mergeTimes = clusterPolicy.mergeTimes
    mergeTimes = agglomerativeClustering.ucmTransform(mergeTimes)


    seg = agglomerativeClustering.result()

testMala()