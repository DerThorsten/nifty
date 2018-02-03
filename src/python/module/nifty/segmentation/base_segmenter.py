

# TODO
class SegmenterFromOversegmentation(object):
    pass


class SegmenterFromCosts(object):
    def __call__(self, graph, costs, node_sizes=None, edge_sizes=None, **kwargs):
        return self._segmentation_impl(graph, costs, node_sizes, edge_sizes, **kwargs)
