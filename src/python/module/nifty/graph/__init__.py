# sphinx_gallery_thumbnail_number = 4
from __future__ import absolute_import
from . import _graph as __graph
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

for key in __graph.__dict__.keys():
    try:
        __graph.__dict__[key].__module__='nifty.graph'
    except:
        pass
    __all__.append(key)


UndirectedGraph.__module__ = "nifty.graph"


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





def undirectedGraph(numberOfNodes):
    return UndirectedGraph(numberOfNodes)

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





def drawGraph(graph, method='spring'):

    import networkx

    G=networkx.Graph()
    for node in graph.nodes():
        G.add_node(node)

    #uvIds = graph.uvIds()
    #for i in range(uvIds.shape[0]):
    #    u,v = uvIds[i,:]
    #    G.add_edge(u,v)
    for edge in graph.edges():
        u,v = graph.uv(edge)
        G.add_edge(u,v)

    nodeLabels = dict()

    for node in graph.nodes():
        nodeLabels[node] = str(node)
    if method == 'spring':
        networkx.draw_spring(G,labels=nodeLabels)
    else:
        networkx.draw(G, lables=nodeLabels)