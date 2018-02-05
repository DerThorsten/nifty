import nifty.graph.rag as nrag
from .base import ProblemExtractor


# TODO
class MeanAffinitiyMapFeatures(ProblemExtractor):
    def __init__(self, statisric='mean'):
        pass

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 4


# TODO
class MeanBoundaryMapFeatures(ProblemExtractor):
    def __init__(self, statistic='mean'):
        pass

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 3
        features = nrag.accumulateEdgeStandartFeatures(self.rag, input_)   # TODO
        return features[:, self.stat_index]
