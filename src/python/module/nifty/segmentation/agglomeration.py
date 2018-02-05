import numpy as np
from .base import Segmenter

import nifty.graph.agglo as nagglo


class AgglomerationBase(Segmenter):
    def __init__(self, use_sizes=False):
        self.use_sizes = use_sizes

    def _segmentation_impl(self, graph, costs, node_sizes=None, edge_sizes=None):
        if self.use_sizes:
            assert node_sizes is not None and edge_sizes is not None
            node_sizes_ = node_sizes
            edge_sizes_ = edge_sizes
        else:
            node_sizes_ = np.ones(graph.numberOfNodes)
            edge_sizes_ = np.ones(graph.numberOfEdges)
        policy = self._cluster_policy(graph=graph,
                                      edge_features=costs,
                                      edge_sizes=edge_sizes_,
                                      node_sizes=node_sizes_)
        clustering = nagglo.agglomerativeClustering(policy)
        clustering.run()
        node_labels = clustering.result()
        return node_labels


class MalaClustering(AgglomerationBase):
    def __init__(self, threshold, with_sizes=False, size_regularizer=None, **super_kwargs):
        self.threshold = threshold
        self.size_regularizer = 0.5 if size_regularizer is None else size_regularizer
        super(MalaClustering, self).__init__(with_sizes=with_sizes)

    def _cluster_policy(self, graph, edge_features, node_sizes, edge_sizes):
        return nagglo.malaClusterPolicy(graph=graph,
                                        edgeIndicators=edge_features,
                                        nodeSizes=node_sizes,
                                        edgeSizes=edge_sizes,
                                        threshold=self.threshold,
                                        size_regularizer=self.size_regularizer)


class AgglomerativeClustering(AgglomerationBase):
    def __init__(self, n_target_clusters, with_sizes=True, size_regularizer=None):
        self.n_target_clusters = n_target_clusters
        self.size_regularizer = 0.5 if size_regularizer is None else size_regularizer
        super(AgglomerativeClustering, self).__init__(with_sizes=with_sizes)

    def cluster_policy(self, graph, edge_features, node_sizes, edge_sizes):
        return nagglo.edgeWeightedClusterPolicy(graph=graph,
                                                edgeIndicators=edge_features,
                                                nodeSizes=node_sizes,
                                                edgeSizes=edge_sizes,
                                                numberOfNodesStop=self.n_target_clusters,
                                                sizeRegularizer=self.size_regularizer)
