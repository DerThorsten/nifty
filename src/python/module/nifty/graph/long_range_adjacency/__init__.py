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


# TODO
def longRangeAdjacency(labels, numberOfThreads=-1, serialization=None):
    pass
