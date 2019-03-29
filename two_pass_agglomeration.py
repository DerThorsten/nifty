from concurrent import futures
import numpy as np

import nifty
import nifty.graph.rag as nrag


def dummy_agglomerator(affs, offsets, previous_segmentation=None,
                       previous_edge=None, previous_weights=None, return_state=False,
                       **parameters):
    pass


def make_checkorboard(blocking):
    """
    """
    blocks1 = [0]
    blocks2 = []
    all_blocks = [0]

    def recurse(current_block, insert_list):
        other_list = blocks1 if insert_list is blocks2 else blocks2
        for dim in range(3):
            ngb_id = blocking.getNeighborId(current_block, dim, False)
            if ngb_id != -1:
                if ngb_id not in all_blocks:
                    insert_list.append(ngb_id)
                    all_blocks.append(ngb_id)
                    recurse(ngb_id, other_list)

    recurse(0, blocks2)
    all_blocks = blocks1 + blocks2
    expected = set(range(blocking.numberOfBlocks))
    assert len(all_blocks) == len(expected), "%i, %i" % (len(all_blocks), len(expected))
    assert len(set(all_blocks) - expected) == 0
    assert len(blocks1) == len(blocks2), "%i, %i" % (len(blocks1), len(blocks2))
    return blocks1, blocks2


# find segments in segmentation that originate from seeds
def get_assignments(segmentation, seeds):
    seed_ids, seed_indices = np.unique(seeds, return_index=True)
    # 0 stands for unseeded
    seed_ids, seed_indices = seed_ids[1:], seed_indices[1:]
    seg_ids = segmentation.ravel()[seed_indices]
    assignments = np.concatenate([seed_ids[:, None], seg_ids[:, None]], axis=1)
    return assignments


def two_pass_agglomeration(affinities, offsets, agglomerator,
                           block_shape, halo, n_threads):
    """ Run two-pass agglommeration
    """
    assert affinities.ndim == 4
    assert affinities.shape[0] == len(offsets)
    assert callable(agglomerator)
    assert len(block_shape) == len(halo) == 3

    shape = affinities.shape[1:]
    blocking = nifty.tools.blocking([0, 0, 0], list(shape), list(block_shape))
    block_size = np.prod(block_shape)

    segmentation = np.zeros(shape, dtype='uint64')

    # calculations for pass 1:
    #
    def pass1(block_id):
        # TODO we could already add some halo here, that might help to make results more consistento

        # load the affinities from the current block
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        aff_bb = (slice(None),) + bb
        affs = affinities[aff_bb]

        # TODO need to define the api for the agglomerator
        # get the segmentation from our agglomeration function
        # NOTE we could also generate the state with the agglomerator directly
        # in that case, we would nee to add up the id_offset to the uv-ids
        seg, state = agglomerator(affs, offsets, return_state=True)

        # offset the segmentation with the lowest block coordinate to
        # make segmentation ids unique
        id_offset = block_id * block_size
        seg += id_offset
        uvs, weights = state
        uvs += id_offset

        # write out the segmentation
        segmentation[bb] = seg

        # compute the state of the segmentation and return it
        return uvs, weights

    # get blocks corresponding to the two checkerboard colorings
    blocks1, blocks2 = make_checkorboard(blocking)

    # TODO use threadpool once this is debugged
    # with futures.ThreadPoolExecutor(n_threads) as tp:
    #   tasks = [tp.submit(pass1, block_id) for block_id in blocks1]
    #   results = [t.result() for t in tasks]
    results = [pass1(block_id) for block_id in blocks1]

    # combine results and build graph corresponding to it
    uvs = np.concatenate([res[0] for res in results], axis=0)
    n_labels = int(uvs.max()) + 1
    graph = nifty.graph.undirectedGraph(n_labels)
    graph.insertEdges(uvs)
    weights = np.concatenate([res[1] for res in results], axis=0)
    assert len(uvs) == len(weights)

    # calculations for pass 2:
    #
    def pass2(block_id):
        # load affinities and segmentation from pass1 from the current block with halo
        block = blocking.getBlockWithHalo(block_id, list(halo))
        bb = tuple(slice(beg, end) for beg, end in zip(block.outerBlock.begin, block.outerBlock.end))
        seg = segmentation[bb]
        aff_bb = (slice(None),) + bb
        affs = affinities[aff_bb]

        # get the state of the segmentation from pass 1
        # TODO maybe there is a better option than doing this with the rag
        rag = nrag.gridRag(seg, numberOfLabels=int(seg.max() + 1), numberOfThreads=1)
        prev_uv_ids = rag.uvIds()
        prev_uv_ids = prev_uv_ids[(prev_uv_ids != 0).all(axis=1)]
        edge_ids = graph.findEdges(prev_uv_ids)
        assert len(edge_ids) == len(prev_uv_ids), "%i, %i" % (len(edge_ids), len(prev_uv_ids))

        # TODO for some reason we can get edges here that are not part of the serialized state
        # I don't fully get why, but it means that we have seeds from the different pass 1
        # blocks touching
        # for now, we just get rid of these edges
        # assert (edge_ids != -1).all()
        valid_edges = edge_ids == -1
        edge_ids = edge_ids[valid_edges]
        prev_uv_ids= prev_uv_ids[valid_edges]
        prev_weights = weights[edge_ids]
        assert len(prev_uv_ids) == len(prev_weights)

        # call the agglomerator with state
        new_seg = agglomerator(affs, offsets, previous_segmentation=seg,
                               previous_edges=prev_uv_ids, previous_weights=prev_weights)

        # offset the segmentation with the lowest block coordinate to
        # make segmentation ids unique
        id_offset = block_id * block_size
        new_seg += id_offset

        # find the assignments to seed ids
        assignments = get_assignments(new_seg, seg)

        # write out the segmentation
        inner_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlock.begin, block.innerBlock.end))
        local_bb = tuple(slice(beg, end) for beg, end in zip(block.innerBlockLocal.begin, block.innerBlockLocal.end))
        segmentation[inner_bb] = new_seg[local_bb]

        return assignments

    # TODO use threadpool once this is debugged
    # with futures.ThreadPoolExecutor(n_threads) as tp:
    #     tasks = [tp.submit(pass2, block_id) for block_id in blocks2]
    #     results = [t.result() for t in tasks]
    results = [pass2(block_id) for block_id in blocks2]
    assignments = np.concatenate(results)

    # get consistent labeling with union find
    n_labels = int(segmentation.max()) + 1
    ufd = nifty.ufd.ufd(n_labels)
    ufd.merge(assignments)
    labeling = ufd.elementLabeling()

    segmentation = nifty.tools.take(labeling, segmentation)
    return segmentation
