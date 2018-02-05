from .base import Oversegmenter


# TODO implement the classic distance trafo watershed
class DTWatershed(Oversegmenter):
    def __init__(self):
        pass

    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 3


# TODO implement the long-range affinity based watershed
class LRAffinityWatershed(Oversegmenter):
    def __init__(self):
        pass

    def _oversegmentation_impl(self, input_):
        assert input_.ndim == 4
