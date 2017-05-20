from __future__ import absolute_import
from ._graph import *

from .. import Configuration
from . import optimization

from . optimization import multicut
from . optimization import lifted_multicut
from . optimization import mincut


# import optimization.multicut as multicut
# import optimization.lifted_multicut as lifted_multicut
# import optimization.mincut as mincut
# from . import rag

import numpy
from functools import partial
import types
import sys

__all__ = []

for key in _graph.__dict__.keys():
    __all__.append(key)


ilpSettings = multicut.ilpSettings





# multicut objective
UndirectedGraph.MulticutObjective                     = multicut.MulticutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph                  = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MulticutObjective = multicut.MulticutObjectiveEdgeContractionGraphUndirectedGraph


UndirectedGraph.MincutObjective                     = mincut.MincutObjectiveUndirectedGraph
UndirectedGraph.EdgeContractionGraph                  = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MincutObjective = mincut.MincutObjectiveEdgeContractionGraphUndirectedGraph


# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph



if sys.version_info >= (1, 0):


    class EdgeContractionGraphCallback(EdgeContractionGraphCallbackImpl):
        def __init__(self):
            super(EdgeContractionGraphCallback, self).__init__()

            try:
                self.contractEdgeCallback = self.contractEdge
            except AttributeError:
                pass

            try:
                self.mergeEdgesCallback = self.mergeEdges
            except AttributeError:
                pass

            try:
                self.mergeNodesCallback = self.mergeNodes
            except AttributeError:
                pass

            try:
                self.contractEdgeDoneCallback = self.contractEdgeDone
            except AttributeError:
                pass
else:

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
EdgeContractionGraphCallback = EdgeContractionGraphCallback

def edgeContractionGraph(g, callback):
    Ecg = g.__class__.EdgeContractionGraph
    ecg = Ecg(g, callback)
    return ecg




