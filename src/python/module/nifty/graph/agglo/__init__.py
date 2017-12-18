from __future__ import absolute_import
from . import _agglo as __agglo
from ._agglo import *

import numpy

__all__ = []
for key in __agglo.__dict__.keys():
    __all__.append(key)
    try:
        __agglo.__dict__[key].__module__='nifty.graph.agglo'
    except:
        pass

from ...tools import makeDense as __makeDense


# def fixationClusterPolicy(graph, 
#     mergePrios=None,
#     notMergePrios=None,
#     edgeSizes=None,
#     isLocalEdge=None,
#     updateRule0="smooth_max",
#     updateRule1="smooth_max",
#     p0=float('inf'),
#     p1=float('inf'),
#     zeroInit=False):
    
#     if isLocalEdge is None:
#         raise RuntimeError("`isLocalEdge` must not be none")

#     if mergePrios is None and if notMergePrios is  None:
#         raise RuntimeError("`mergePrios` and `notMergePrios` cannot be both None")

#     if mergePrio is None:
#         nmp = notMergePrios.copy()
#         nmp -= nmp.min()
#         nmp /= nmp.max()
#         mp = 1.0 = nmp
#     elif notMergePrios is None:
#         mp = notMergePrios.copy()
#         mp -= mp.min()
#         mp /= mp.max()
#         nmp = 1.0 = mp
#     else:
#         mp = mergePrios
#         nmp = notMergePrios

#     if edgeSizes is None:
#         edgeSizes = numpy.ones(graph.edgeIdUpperBound+1)




#     if(updateRule0 == "histogram_rank" and updateRule1 == "histogram_rank"):
#         return nifty.graph.agglo.rankFixationClusterPolicy(graph=graph,
#             mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         q0=p0, q1=p1, zeroInit=zeroInit)
#     elif(updateRule0 in ["smooth_max","generalized_mean"] and updateRule1 in ["smooth_max","generalized_mean"]):
        

#         return  nifty.graph.agglo.generalizedMeanFixationClusterPolicy(graph=g,
#                         mergePrios=mp, notMergePrios=nmp,
#                         edgeSizes=edgeSizes, isMergeEdge=isLocalEdge,
#                         p0=p0, p1=p1, zeroInit=zeroInit)

       













def sizeLimitClustering(graph, nodeSizes, minimumNodeSize, 
                        edgeIndicators=None,edgeSizes=None, 
                        sizeRegularizer=0.001, gamma=0.999,
                        makeDenseLabels=False):

    s = graph.edgeIdUpperBound + 1

    def rq(data):
        return numpy.require(data, 'float32')

    nodeSizes  = rq(nodeSizes)

    if edgeIndicators is None:
        edgeIndicators = numpy.ones(s,dtype='float32')
    else:
        edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)



    cp =  minimumNodeSizeClusterPolicy(graph, edgeIndicators=edgeIndicators, 
                                              edgeSizes=edgeSizes,
                                              nodeSizes=nodeSizes,
                                              minimumNodeSize=float(minimumNodeSize),
                                              sizeRegularizer=float(sizeRegularizer),
                                              gamma=float(gamma))

    agglo = agglomerativeClustering(cp)

    agglo.run()
    labels = agglo.result()

    if makeDenseLabels:
        labels = __makeDense(labels)

    return labels;




def ucmFeatures(graph, edgeIndicators, edgeSizes, nodeSizes, 
                sizeRegularizers = numpy.arange(0.1,1,0.1) ):
    
    def rq(data):
        return numpy.require(data, 'float32')
 
    edgeIndicators = rq(edgeIndicators)

    if edgeSizes is None:
        edgeSizes = numpy.ones(s,dtype='float32')
    else:
        edgeSizes = rq(edgeSizes)


    if nodeSizes is None:
        nodeSizes = numpy.ones(s,dtype='float32')
    else:
        nodeSizes = rq(nodeSizes)

    fOut = []
    # policy
    for sr in sizeRegularizers:

        sr = float(sr)
        cp = edgeWeightedClusterPolicyWithUcm(graph=graph, edgeIndicators=edgeIndicators,
                edgeSizes=edgeSizes, nodeSizes=nodeSizes, sizeRegularizer=sr)


        agglo = agglomerativeClustering(cp)



        hA = agglo.runAndGetDendrogramHeight()[:,None]
        hB = agglo.ucmTransform(cp.edgeIndicators)[:,None]

        fOut.extend([hA,hB])

    return numpy.concatenate(fOut, axis=1)

