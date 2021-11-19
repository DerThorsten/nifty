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


import numpy as np

def accumulate_affinities_mean_and_length(affinities, offsets, labels, graph=None,
                                          affinities_weights=None,
                                          offset_weights=None,
                                          ignore_label=None, number_of_threads=-1):
    """

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
        graph = gridRag(labels)


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

    Parameters
    ----------
    affinities: offset channels expected to be the first one
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