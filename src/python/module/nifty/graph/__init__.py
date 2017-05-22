from __future__ import absolute_import
from ._graph import *

from .. import Configuration
from . import optimization

from . optimization import multicut
from . optimization import lifted_multicut
from . optimization import mincut

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
UndirectedGraph.EdgeContractionGraph                = EdgeContractionGraphUndirectedGraph
EdgeContractionGraphUndirectedGraph.MincutObjective = mincut.MincutObjectiveEdgeContractionGraphUndirectedGraph

# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph



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

def edgeContractionGraph(g, callback):
    Ecg = g.__class__.EdgeContractionGraph
    ecg = Ecg(g, callback)
    return ecg




def undirectedGridGraph(shape, simpleNh=True):
    if not simpleNh:
        raise RuntimeError("currently only simpleNh is implemented")
    s = [int(s) for s in shape]
    if(len(s) == 2):
        return UndirectedGridGraph2DSimpleNh(s)
    elif(len(s) == 3):
        return UndirectedGridGraph3DSimpleNh(s)
    else:
        raise RuntimeError("currently only 2D and 3D grid graph is exposed to python")

