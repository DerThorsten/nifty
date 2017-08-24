from __future__ import absolute_import
from . import _long_range_adjacency as __long_range_adjacency
from ._long_range_adjacency import *
from .. import Configuration

__all__ = []
for key in __long_range_adjacency.__dict__.keys():
    try:
        __long_range_adjacency.__dict__[key].__module__ = 'nifty.graph.long_range_adjacency'
    except:
        pass

    __all__.append(key)


def longRangeAdjacency(labels, longRange, numberOfThreads=-1, serialization=None):
    assert labels.ndim == 3
    if serialization is None:
        numberOfLabels = labels.max() + 1
        return explicitLabelsLongRangeAdjacency(labels, longRange, numberOfLabels, numberOfThreads)
    else:
        return explicitLabelsLongRangeAdjacency(labels, serialization)


if Configuration.WITH_HDF5:

    def longRangeAdjacencyHDF5(labels, longRange, numberOfLabels, numberOfThreads=-1, serialization=None):
        if serialization is None:
            return hdf5LabelsLongRangeAdjacency(labels,  longRange, numberOfLabels, numberOfThreads)
        else:
            return hdf5LabelsLongRangeAdjacency(labels, serialization)
