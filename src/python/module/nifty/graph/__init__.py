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


def run_label_propagation(graph, edge_values=None, nb_iter=1, local_edges=None, size_constr=-1,
                          nb_threads=-1):
    """
    This function can be useful to obtain superpixels (alternative to WS superpixels for example).

    The usual label propagation algorithm (https://en.wikipedia.org/wiki/Label_propagation_algorithm) iterates
        over nodes of the graph in a random order: for every iteration and selected node u,
        the algorithm assigns u to the label occurring with the highest frequency among its neighbours
        (if there are multiple highest frequency labels, it selects a label at random).
        This process can be repeated multiple times (`nb_iter`) until the algorithm converges to a set of labels.


    This generalized implementation also supports signed edge values, so that node labels are not assigned to the neighboring
        label with higher frequency, but to the neighboring label with the highest positive edge interaction.
        By default, all edge values have weight +1 and the standard label propagation algorithm is performed.

        For example, a node with the following five-nodes neighborhood:

         - neighbor_1_label = 1, edge_weight = +2
         - neighbor_2_label = 1, edge_weight = +5
         - neighbor_3_label = 1, edge_weight = -2
         - neighbor_4_label = 2, edge_weight = -5
         - neighbor_5_label = 3, edge_weight = +5

        will be randomly assigned to label 1 or 3 (given they have equal maximum attraction +5).

    :param graph:       undirected graph
    :param edge_values: Optional signed edge weights. By default, all edges have equal weight +1 and the standard
                        label propagation algorithm is performed .
    :param nb_iter:     How many label propagation iterations to perform
                        (one iteration = one loop over all the nodes of the graph)
    :param local_edges: Boolean array indicating which edges are local edges in the graph. If specified, then the
                        algorithm proceeds as following: any given node can be assigned to the label of
                        a neighboring cluster only if this cluster has at least one local edge connection to the node.
    :param size_constr: Whether or not to set a maximum size for the final clusters.
                        The default value is -1 and no size constraint is applied.
    :param nb_threads:  When multiple threads are used, multiple nodes are processed in parallel.

    :return: Newly assigned node labels
    """
    nb_edges = graph.numberOfEdges
    edge_values = numpy.ones((nb_edges,), dtype="float32") if edge_values is None else edge_values
    assert edge_values.shape[0] == nb_edges

    if local_edges is not None:
        assert edge_values.shape == local_edges.shape
        local_edges = numpy.require(local_edges, dtype='bool')
    else:
        local_edges = numpy.ones_like(edge_values).astype('bool')

    # TODO: add support initial node_labels (need to specify initial cluster size)
    nb_nodes = graph.numberOfNodes
    node_labels = numpy.arange(0, nb_nodes)
    # if node_labels is None:
    #     node_labels = numpy.arange(0, nb_nodes)
    #     sizes = numpy.ones((nb_nodes,))
    # else:
    #     raise NotImplementedError()
    node_labels = numpy.require(node_labels, dtype='uint64')

    runLabelPropagation_impl(graph, node_labels, edge_values, local_edges, nb_iter, size_constr, nb_threads)

    return node_labels


import numpy as np
import nifty.graph.rag as nrag

def accumulate_affinities_mean_and_length(affinities, offsets, labels, graph=None,
                                          affinities_weights=None,
                                          offset_weights=None,
                                          ignore_label=None, number_of_threads=-1):
    """
    Features of this function (additional ones compared to other accumulate functions):
      - does not require a RAG but simply a graph and a label image (can include long-range edges)
      - can perform weighted average of affinities depending on given affinitiesWeights
      - ignore pixels with ignore label

    Parameters
    ----------
    affinities: offset channels expected to be the first one
    """
    affinities = np.require(affinities, dtype='float32')

    if affinities_weights is not None:
        assert offset_weights is None, "Affinities weights and offset weights cannot be passed at the same time"
        affinities_weights = np.require(affinities_weights, dtype='float32')

    else:
        affinities_weights = np.ones_like(affinities)
        if offset_weights is not None:
            offset_weights = np.require(offset_weights, dtype='float32')
            for _ in range(affinities_weights.ndim-1):
                offset_weights = np.expand_dims(offset_weights, axis=-1)
            affinities_weights *= offset_weights

    affinities = np.rollaxis(affinities, axis=0, start=len(affinities.shape))
    affinities_weights = np.rollaxis(affinities_weights, axis=0, start=len(affinities_weights.shape))

    offsets = np.require(offsets, dtype='int32')
    assert len(offsets.shape) == 2

    if graph is None:
        graph = nrag.gridRag(labels)


    hasIgnoreLabel = (ignore_label is not None)
    ignore_label = 0 if ignore_label is None else int(ignore_label)

    number_of_threads = -1 if number_of_threads is None else number_of_threads

    edge_indicators_mean, edge_indicators_max, edge_sizes = \
        accumulateAffinitiesMeanAndLength_impl_(
            graph,
            labels.astype('uint64'),
            affinities,
            affinities_weights,
            offsets,
            hasIgnoreLabel,
            ignore_label,
            number_of_threads
        )
    return edge_indicators_mean, edge_sizes


def accumulate_affinities_mean_and_length_inside_clusters(affinities, offsets, labels,
                                                          offset_weights=None,
                                                          ignore_label=None, number_of_threads=-1):
    """
    Similar idea to `accumulate_affinities_mean_and_length`, but accumulates affinities/edge-values for all edges not on
     cut (i.e. connecting nodes in the same cluster)
    """
    affinities = np.require(affinities, dtype='float32')
    affinities = np.rollaxis(affinities, axis=0, start=len(affinities.shape))

    offsets = np.require(offsets, dtype='int32')
    assert len(offsets.shape) == 2

    if offset_weights is None:
        offset_weights = np.ones(offsets.shape[0], dtype='float32')
    else:
        offset_weights = np.require(offset_weights, dtype='float32')

    hasIgnoreLabel = (ignore_label is not None)
    ignore_label = 0 if ignore_label is None else int(ignore_label)

    number_of_threads = -1 if number_of_threads is None else number_of_threads

    edge_indicators_mean, edge_indicators_max, edge_sizes = \
        accumulateAffinitiesMeanAndLengthInsideClusters_impl_(
            labels.astype('uint64'),
            labels.max(),
            affinities,
            offsets,
            offset_weights,
            hasIgnoreLabel,
            ignore_label,
            number_of_threads
        )
    return edge_indicators_mean, edge_sizes