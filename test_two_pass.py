from functools import partial
import numpy as np
import z5py

import nifty.graph.rag as nrag
from cluster_tools.utils.segmentation_utils import mutex_watershed_with_seeds, mutex_watershed
# from cluster_tools.utils.segmentation_utils import compute_grid_graph
from two_pass_agglomeration import two_pass_agglomeration
from cremi_tools.viewer.volumina import view


# TODO this should also take strides and randomize_strides
def compute_state(affs, seg, offsets, n_attractive):

    # with affogato TODO debug this
    # FIXME the uv ids don't make sense!
    # grid_graph = compute_grid_graph(segmentation.shape)
    # uvs, weights, attractive = grid_graph.compute_state_for_segmentation(affs, segmentation, offsets,
    #                                                                      n_attractive_channels=3,
    #                                                                      ignore_label=False)
    # weights[np.logical_not(attractive)] *= -1
    # state = (uvs, weights)

    # with nifty
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max() + 1),
                       numberOfThreads=1)
    uv_ids = rag.uvIds()

    affs_attractive = affs[:n_attractive]
    # -2 corresponds to max value
    weights_attractive = nrag.accumulateAffinityStandartFeatures(rag, affs_attractive, offsets,
                                                                 numberOfThreads=1)[:, -2]

    affs_repulsive = np.require(affs[n_attractive:], requirements='C')
    weights_repulsive = nrag.accumulateAffinityStandartFeatures(rag, affs_repulsive, offsets,
                                                                numberOfThreads=1)[:, -2]

    weights = weights_attractive
    repulsive = weights_repulsive > weights_attractive
    weights[repulsive] = -1*weights_repulsive[repulsive]
    return uv_ids, weights


def mws_agglomerator(affs, offsets, previous_segmentation=None,
                     previous_edges=None, previous_weights=None, return_state=False,
                     strides=None, randomize_strides=True):

    if previous_segmentation is not None:
        assert previous_edges is not None
        assert previous_weights is not None
        assert len(previous_edges) == len(previous_weights), "%i, %i" % (len(previous_edges),
                                                                         len(previous_weights))

        # transform the seed state to what is expected by mutex_watershed_with_seeds
        repulsive = previous_weights < 0
        attractive = np.logical_not(repulsive)
        seed_state = {'attractive': (previous_edges[attractive], previous_weights[attractive]),
                      'repulsive': (previous_edges[repulsive], np.abs(previous_weights[repulsive]))}

        segmentation = mutex_watershed_with_seeds(affs, offsets, seeds=previous_segmentation,
                                                  strides=strides, randomize_strides=randomize_strides,
                                                  seed_state=seed_state)
    else:
        segmentation = mutex_watershed(affs, offsets, strides,
                                       randomize_strides=randomize_strides)

    if return_state:
        state = compute_state(affs, segmentation, offsets, 3)
        return segmentation, state
    return segmentation


def test_tp():
    path = '/home/pape/Work/data/cluster_tools_test_data/test_data.n5'
    aff_key = '/volumes/full_affinities'

    f = z5py.File(path)
    ds_affs = f[aff_key]
    ds_affs.n_threads = 8
    affs = ds_affs[:]

    # affs = affs[:, :10, :256]
    # affs = affs[:, :20, :256]
    print(affs.shape)

    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
               [0, -9, 0], [0, 0, -9],
               [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
               [0, -27, 0], [0, 0, -27]]

    block_shape = [10, 256, 256]
    halo = [2, 32, 32]

    print("Start agglomeration")
    agglomerator = partial(mws_agglomerator, strides=[2, 10, 10], randomize_strides=True)
    seg = two_pass_agglomeration(affs, offsets, agglomerator, block_shape, halo, 4)
    print(seg.shape)

    view([affs[1], seg])


if __name__ == '__main__':
    test_tp()
