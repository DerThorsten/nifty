from __future__ import absolute_import
from . import _long_range_adjacency as __long_range_adjacency
from ._long_range_adjacency import *
from .. import Configuration
from ..rag import gridRag

__all__ = []
for key in __long_range_adjacency.__dict__.keys():
    try:
        __long_range_adjacency.__dict__[key].__module__ = 'nifty.graph.long_range_adjacency'
    except:
        pass

    __all__.append(key)


def longRangeAdjacency(labels, numberOfLabels=None, longRange=None,
                       ignoreLabel=False, numberOfThreads=-1, serialization=None):
    assert labels.ndim == 3
    if serialization is None:
        assert numberOfLabels is not None and longRange is not None
        return explicitLabelsLongRangeAdjacency(labels, longRange, numberOfLabels, ignoreLabel, numberOfThreads)
    else:
        return explicitLabelsLongRangeAdjacency(labels, serialization)


if Configuration.WITH_HDF5:

    def longRangeAdjacencyHDF5(labels, numberOfLabels=None, longRange=None,
                               ignoreLabel=False, numberOfThreads=-1, serialization=None):
        if serialization is None:
            return hdf5LabelsLongRangeAdjacency(labels,  longRange, numberOfLabels, ignoreLabel, numberOfThreads)
        else:
            assert numberOfLabels is not None and longRange is not None
            return hdf5LabelsLongRangeAdjacency(labels, serialization)


