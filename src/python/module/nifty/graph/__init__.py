from __future__ import absolute_import
from ._graph import *
from .. import Configuration

from . import multicut
from . import lifted_multicut
# from . import rag

import numpy
from functools import partial
import types

__all__ = []

for key in _graph.__dict__.keys():
    __all__.append(key)


ilpSettings = multicut.ilpSettings





# multicut objective
UndirectedGraph.MulticutObjective = multicut.MulticutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MulticutObjective = multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph





# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph






class EdgeContractionGraphCallback(EdgeContractionGraphCallbackImpl):
    def __init__(self):
        super(EdgeContractionGraphCallback, self).__init__()


        try:
            self.contractEdgeCallback = types.MethodType(self.contractEdge, self,
                                            EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:
            self.mergeEdgesCallback = types.MethodType(self.mergeEdges, self,
                                            EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:
            self.mergeNodesCallback = types.MethodType(self.mergeNodes, self,
                                        EdgeContractionGraphCallback)
        except AttributeError:
            pass

        try:
            self.contractEdgeDoneCallback = types.MethodType(self.contractEdgeDone, self,
                                        EdgeContractionGraphCallback)
        except AttributeError:
            pass

    #def contractEdgeCallback(self, edge):
    #    pass
    #def contractEdgeDoneCallback(self, edge):
    #    pass

#EdgeContractionGraphCallback.__module__ = "graph"
EdgeContractionGraphCallback = EdgeContractionGraphCallback

def edgeContractionGraph(g, callback):
    Ecg = g.__class__.EdgeContractionGraph
    ecg = Ecg(g, callback)
    return ecg




