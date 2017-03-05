import numpy as np
import nifty
import vigra
import os


def make_random_stacked_seg(shape):
    seg = np.zeros(shape, dtype = 'uint32')
    label = 0
    for z in xrange(shape[0]):
        if z > 0:
            label = np.max(seg[z-1]) + 1
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):
                seg[z,y,x] = label
                if np.random.randint(0,10) > 8:
                    label += 1
    return seg

def check_stacked(seg):
    prev_max = -1
    for z in xrange(seg.shape[0]):
        min_label = seg[z].min()
        max_label = seg[z].max()
        assert min_label == prev_max + 1
        prev_max = max_label

def test_hdf5_serialization():
    seg = make_random_stacked_seg( (20,100,100) )
    check_stacked(seg)
    n_labels = seg.max() + 1

    vigra.writeHDF5(seg, './seg_tmp.h5', 'data')

    label_f = nifty.hdf5.openFile('./seg_tmp.h5')
    labels = nifty.hdf5.Hdf5ArrayUInt32(label_f, 'data')
    print "Labels done"

    rag = nifty.graph.rag.gridRagStacked2DHdf5(labels, n_labels, -1)
    print "Rag done"

    nifty.graph.rag.writeStackedRagToHdf5(rag,'./rag_tmp.h5')
    rag_read = nifty.graph.rag.readStackedRagFromHdf5(labels, n_labels, './rag_tmp.h5')
    print "Read/write done"

    try:
        assert (rag.uvIds() == rag_read.uvIds()).all(), "Test uvs failed"
        assert (rag.numberOfEdges == rag_read.numberOfEdges), "Test nedges failed"
        assert (rag.numberOfNodes == rag_read.numberOfNodes), "Test nnodes failed"
        assert (rag.minMaxLabelPerSlice() == rag_read.minMaxLabelPerSlice()).all(), "Test minmax label failed"
        assert (rag.numberOfNodesPerSlice() == rag_read.numberOfNodesPerSlice()).all(), "Test nnodes slice failed"
        assert (rag.numberOfInSliceEdges() == rag_read.numberOfInSliceEdges()).all(), "Test inslice edges slice failed"
        assert (rag.numberOfInBetweenSliceEdges() == rag_read.numberOfInBetweenSliceEdges()).all(), "Test betweenslice edges slice failed"
        assert (rag.inSliceEdgeOffset() == rag_read.inSliceEdgeOffset()).all(), "Test inslice offset failed"
        assert (rag.betweenSliceEdgeOffset() == rag_read.betweenSliceEdgeOffset()).all(), "Test betweenslice offset failed"
        assert (rag.totalNumberOfInSliceEdges == rag_read.totalNumberOfInSliceEdges), "Test total inslice failed"
        assert (rag.totalNumberOfInBetweenSliceEdges == rag_read.totalNumberOfInBetweenSliceEdges), "Test total between slice failed"

        print "Passed"
    except AssertionError as e:
        os.remove('./rag_tmp.h5')
        os.remove('./seg_tmp.h5')
        raise e
    os.remove('./rag_tmp.h5')
    os.remove('./seg_tmp.h5')


if __name__ == '__main__':
    test_hdf5_serialization()
