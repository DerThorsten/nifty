# sphinx_gallery_thumbnail_number = 4
from __future__ import absolute_import
from . import _graph as __graph
from ._graph import *

from .. import Configuration
from . import opt

from . opt import multicut
from . opt import lifted_multicut
from . opt import mincut
from . opt import minstcut

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

# #minstcut objective
# UndirectedGraph.MinstcutObjective                     = minstcut.MinstcutObjectiveUndirectedGraph
# UndirectedGraph.EdgeContractionGraph                = EdgeContractionGraphUndirectedGraph
# EdgeContractionGraphUndirectedGraph.MinstcutObjective = minstcut.MinstcutObjectiveEdgeContractionGraphUndirectedGraph

# lifted multicut objective
UndirectedGraph.LiftedMulticutObjective = lifted_multicut.LiftedMulticutObjectiveUndirectedGraph


def randomGraph(numberOfNodes, numberOfEdges):
    g = UndirectedGraph(numberOfNodes)

    uv = numpy.random.randint(low=0, high=numberOfNodes-1, size=numberOfEdges*2)
    uv = uv.reshape([-1,2])

    where = numpy.where(uv[:,0]!=uv[:,1])[0]
    uv = uv[where,:]

    g.insertEdges(uv)
    while( g.numberOfEdges < numberOfEdges):
        u,v = numpy.random.randint(low=0, high=numberOfNodes-1, size=2)
        if u != v:
            g.insertEdge(int(u),int(v))
    return g


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

gridGraph = undirectedGridGraph


def undirectedLongRangeGridGraph(shape, offsets, edge_mask=None,
                                 offsets_probabilities=None):
    """
    :param edge_mask: Boolean array (4D) indicating which edge connections should be introduced in the graph.
    :param offsets_probabilities: Probability that a type of neighboring connection is introduced as edge in the graph.
                Cannot be used at the same time with edge_mask
    """
    offsets = numpy.require(offsets, dtype='int64')
    shape = list(shape)
    if len(shape) == 2:
        G = UndirectedLongRangeGridGraph2D
    elif len(shape) == 3:
        G = UndirectedLongRangeGridGraph3D
    else:
        raise RuntimeError("wrong dimension: undirectedLongRangeGridGraph is only implemented for 2D and 3D")

    if edge_mask is not None:
        assert offsets_probabilities is None, "Edge mask and offsets probabilities cannot be used at the same time."
        assert edge_mask.ndim == len(shape) + 1
        assert edge_mask.dtype == numpy.dtype('bool')
        assert edge_mask.shape[0] == offsets.shape[0]
        edge_mask = numpy.rollaxis(edge_mask, axis=0, start=len(shape) + 1)

        useEdgeMask = True
    elif offsets_probabilities is not None:
        offsets_probabilities = numpy.require(offsets_probabilities, dtype='float32')
        assert offsets_probabilities.shape[0] == offsets.shape[0]
        assert (offsets_probabilities.min() >= 0.0) and (offsets_probabilities.max() <= 1.0)

        # Randomly sample some edges to add to the graph:
        edge_mask = []
        for off_prob in offsets_probabilities:
            edge_mask.append(numpy.random.random(shape) <= off_prob)
        edge_mask = numpy.stack(edge_mask, axis=-1)

        useEdgeMask = True
    else:
        # Create an empty edge_mask (anyway it won't be used):
        edge_mask = numpy.empty(tuple([1 for _ in range(len(shape) + 1)]), dtype='bool')
        useEdgeMask = False


    return G(shape=shape, offsets=offsets, edgeMask=edge_mask,
             useEdgeMask=useEdgeMask)

longRangeGridGraph = undirectedLongRangeGridGraph


def drawGraph(graph, method='spring'):
    import networkx

    G = networkx.Graph()
    for node in graph.nodes():
        G.add_node(node)

    for edge in graph.edges():
        u, v = graph.uv(edge)
        G.add_edge(u, v)

    nodeLabels = {node: str(node) for node in graph.nodes()}

    if method == 'spring':
        networkx.draw_spring(G, labels=nodeLabels)
    else:
        networkx.draw(G, lables=nodeLabels)


def run_label_propagation(graph, edge_values, nb_iter=1, node_labels=None, local_edges=None, size_constr=-1,
                          nb_threads=-1):
    print("Start")
    if local_edges is not None:
        assert edge_values.shape == local_edges.shape
        local_edges = numpy.require(local_edges, dtype='bool')
    else:
        local_edges = numpy.ones_like(edge_values).astype('bool')

    nb_nodes = graph.numberOfNodes
    if node_labels is None:
        node_labels = numpy.arange(0, nb_nodes)
    else:
        raise NotImplementedError("Deduce size of initial clusters!")
        assert edge_values.shape == node_labels.shape
    node_labels = numpy.require(node_labels, dtype='uint64')
    sizes = numpy.ones((nb_nodes,))

    runLabelPropagation_impl(graph, node_labels, edge_values, local_edges, nb_iter, size_constr, nb_threads)

    return node_labels