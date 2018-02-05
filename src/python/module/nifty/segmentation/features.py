import nifty.graph.rag as nrag
from .base import ProblemExtractor


# TODO
STAT_TO_INDEX = {'mean': 0}


# TODO
class MeanAffinitiyMapFeatures(ProblemExtractor):
    def __init__(self, statistic='mean'):
        assert statistic in STAT_TO_INDEX
        self.stat_index = STAT_TO_INDEX[statistic]

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 4


# TODO
class MeanBoundaryMapFeatures(ProblemExtractor):
    def __init__(self, statistic='mean'):
        assert statistic in STAT_TO_INDEX
        self.stat_index = STAT_TO_INDEX[statistic]

    def _compute_edge_probabilities(self, input_, fragments=None):
        assert input_.ndim == 3
        features = nrag.accumulateEdgeStandartFeatures(self.rag, input_)   # TODO
        return features[:, self.stat_index]
