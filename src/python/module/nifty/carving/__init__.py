from ._carving import *


def carvingSegmenter(rag, edgeWeights, sortEdges=True):
    ndim = len(rag.shape)
    if ndim == 2:
        return CarvingSegmenterRag2D(rag, edgeWeights, sortEdges)
    else:
        return CarvingSegmenterRag3D(rag, edgeWeights, sortEdges)
