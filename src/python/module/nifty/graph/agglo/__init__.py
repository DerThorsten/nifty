from _agglo import *

__all__ = []
for key in _agglo.__dict__.keys():
    __all__.append(key)


from ...tools import makeDense as __makeDense


def sizeLimitClustering(graph, nodeSizes, minimumNodeSize, 
                        edgeWeights=None,edgeSizes=None, 
                        sizeRegularizer=0.001, gamma=0.999,
                        makeDenseLabels=False):

    s = graph.edgeIdUpperBound() + 1

    def rq(data):
        return numpy.require(data, 'float32');

    nodeSizes  = rq(nodeSizes)

    if edgeWeights is None:
        edgeWeights = numpy.ones(s,dtype='float32')
    else:
        edgeWeights = rq(edgeWeights)

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

    cp.run()
    labels = labels.run()

    if makeDenseLabels:
        labeles = __makeDense(labels)

    return labels;

