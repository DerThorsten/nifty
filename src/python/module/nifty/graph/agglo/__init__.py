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

