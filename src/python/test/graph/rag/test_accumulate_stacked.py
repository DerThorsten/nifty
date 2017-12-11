import unittest
import os
from shutil import rmtree

import numpy as np
import nifty
import nifty.graph.rag as nrag


try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False


class TestAccumulateStacked(unittest.TestCase):
    # shape = (10, 256, 256)
    shape = (5, 10, 10)

    @staticmethod
    def make_labels(shape):
        labels = np.zeros(shape, dtype='uint32')
        label = 0
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    labels[z, y, x] = label
                    if np.random.random() > .95:
                        have_increased = True
                        label += 1
                    else:
                        have_increased = False
            if not have_increased:
                label += 1
        return labels

    def setUp(self):
        self.data = np.random.random(size=self.shape).astype('float32')
        self.labels = self.make_labels(self.shape)
        self.n_labels = self.labels.max() + 1
        self.tmp_dir = './tmp'
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            rmtree(self.tmp_dir)

    def accumulation_in_core_test(self, accumulation_function):
        rag = nrag.gridRagStacked2D(self.labels,
                                    numberOfLabels=self.n_labels,
                                    numberOfThreads=1)
        n_edges_xy = rag.numberOfInSliceEdges
        n_edges_z  = rag.numberOfInBetweenSliceEdges

        # test complete accumulation
        print("Complete Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  numberOfThreads=1)
        self.assertEqual(len(feats_xy), rag.numberOfEdges)
        self.assertEqual(len(feats_z), rag.numberOfEdges)
        # TODO test that things are non-trivial

        # test xy-feature accumulation
        print("Complete XY Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  keepXYOnly=True,
                                                  numberOfThreads=-1)
        self.assertEqual(len(feats_xy), rag.numberOfEdges)
        self.assertEqual(len(feats_z), 1)

        # test z-feature accumulation for all 3 directions
        print("Complete Z Accumulations ...")
        for z_direction in (0, 1, 2):
            feats_xy, feats_z = accumulation_function(rag,
                                                      self.data,
                                                      keepXYOnly=True,
                                                      numberOfThreads=-1)
            self.assertEqual(len(feats_xy), 1)
            self.assertEqual(len(feats_z), rag.numberOfEdges)

    def test_standard_features_in_core(self):
        self.accumulation_in_core_test(nrag.accumulateEdgeStandardFeatures)

    # FIXME this segfaults in fastfilter wrapper
    def _test_features_from_filters_in_core(self):
        self.accumulation_in_core_test(nrag.accumulateEdgeFeatresFromFilters)


if __name__ == '__main__':
    unittest.main()
