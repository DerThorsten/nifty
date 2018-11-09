from __future__ import print_function
import unittest
import os
import numpy as np
import nifty
import nifty.graph.rag as nrag
import vigra


class TestHdf5Serialization(unittest.TestCase):

    def setUp(self):
        self.rag_path = './rag_tmp.h5'
        self.seg_path = './seg_tmp.h5'

    def tearDown(self):
        if os.path.exists(self.rag_path):
            os.remove(self.rag_path)
        if os.path.exists(self.seg_path):
            os.remove(self.seg_path)

    @staticmethod
    def make_random_stacked_seg(shape):
        seg = np.zeros(shape, dtype = 'uint32')
        label = 0
        for z in range(shape[0]):
            if z > 0:
                label = np.max(seg[z-1]) + 1
            for y in range(shape[1]):
                for x in range(shape[2]):
                    seg[z, y, x] = label
                    if np.random.randint(0,10) > 8:
                        label += 1
        return seg

    @staticmethod
    def check_stacked(seg):
        prev_max = -1
        for z in range(seg.shape[0]):
            min_label = seg[z].min()
            max_label = seg[z].max()
            assert min_label == prev_max + 1
            prev_max = max_label

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_hdf5_serialization(self):
        import nifty.hdf5 as nh5
        seg = self.make_random_stacked_seg((20, 100, 100))
        self.check_stacked(seg)
        n_labels = seg.max() + 1

        vigra.writeHDF5(seg, self.seg_path, 'data')

        label_f = nh5.openFile(self.seg_path)
        labels = nh5.Hdf5ArrayUInt32(label_f, 'data')

        rag = nrag.gridRagStacked2DHdf5(labels, n_labels, -1)

        nifty.graph.rag.writeStackedRagToHdf5(rag, self.rag_path)
        rag_read = nrag.readStackedRagFromHdf5(labels, n_labels, self.rag_path)

        self.assertTrue((rag.uvIds() == rag_read.uvIds()).all())
        self.assertTrue((rag.numberOfEdges == rag_read.numberOfEdges))
        self.assertTrue((rag.numberOfNodes == rag_read.numberOfNodes))
        self.assertTrue((rag.minMaxLabelPerSlice() == rag_read.minMaxLabelPerSlice()).all())
        self.assertTrue((rag.numberOfNodesPerSlice() == rag_read.numberOfNodesPerSlice()).all())
        self.assertTrue((rag.numberOfInSliceEdges() == rag_read.numberOfInSliceEdges()).all())
        self.assertTrue((rag.numberOfInBetweenSliceEdges() == rag_read.numberOfInBetweenSliceEdges()).all())
        self.assertTrue((rag.inSliceEdgeOffset() == rag_read.inSliceEdgeOffset()).all())
        self.assertTrue((rag.betweenSliceEdgeOffset() == rag_read.betweenSliceEdgeOffset()).all())
        self.assertTrue((rag.totalNumberOfInSliceEdges == rag_read.totalNumberOfInSliceEdges))
        self.assertTrue((rag.totalNumberOfInBetweenSliceEdges == rag_read.totalNumberOfInBetweenSliceEdges))


if __name__ == '__main__':
    unittest.main()
