import os
import unittest
import numpy as np

try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

import nifty.graph.long_range_adjacency as nlr


# FIXME multi-threading is broken
class TestLongRangeAdjacency(unittest.TestCase):

    def generate_toy_data(self):
        seg = np.array(
            [[[1, 1, 1],
              [1, 1, 2],
              [1, 2, 3]],
             [[4, 4, 4],
              [4, 5, 5],
              [4, 5, 5]],
             [[6, 6, 6],
              [6, 7, 7],
              [6, 7, 7]]], dtype='uint32'
        )
        return seg, 8

    def generate_toy_affinities(self):
         affs = np.ones((1, 3, 3, 3), dtype='float32')
         affs[:, -1] = 0
         return affs

    def generate_random_data(self):
        seg = np.zeros((100, 100, 100), dtype='uint32')
        current_label = 0
        for z in range(seg.shape[0]):
            for y in range(seg.shape[1]):
                for x in range(seg.shape[2]):
                    seg[z, y, x] = current_label
                    if np.random.rand() > .95:
                        current_label += 1
            current_label += 1
        n_labels = int(seg.max())+ 1
        return seg, n_labels

    def generate_random_affinities(self, long_range):
        return np.random.random(size=(long_range - 1, 100, 100, 100)).astype('float32')

    def checks_toy_data(self, lr):
        self.assertEqual(lr.numberOfNodes, 8)
        self.assertEqual(lr.numberOfEdges, 4)
        uvs = lr.uvIds()
        self.assertEqual(uvs[0].tolist(), [1, 6])
        self.assertEqual(uvs[1].tolist(), [1, 7])
        self.assertEqual(uvs[2].tolist(), [2, 7])
        self.assertEqual(uvs[3].tolist(), [3, 7])

    def test_toy_data(self):
        seg, n_labels = self.generate_toy_data()
        lr = nlr.longRangeAdjacency(seg, 2, n_labels)
        self.checks_toy_data(lr)

    @unittest.skipUnless(WITH_H5PY, "Need h5py")
    def test_serialization(self):
        seg = self.generate_toy_data()
        lr = nlr.longRangeAdjacency(seg, 2, 1,
                                    numberOfThreads=1)
        try:
            with h5py.File('./tmp.h5') as f:
                f.create_dataset('data', data=lr.serialize())
                serialization = f['data'][:]
            lr_ = nlr.longRangeAdjacency(seg, 2, serialization=serialization)
            self.checks_toy_data(lr_)
        finally:
            os.remove('./tmp.h5')

    def test_random_data(self):
        seg, n_labels = self.generate_random_data()
        lr = nlr.longRangeAdjacency(seg, 4, n_labels,
                                    numberOfThreads=1)
        self.assertEqual(lr.numberOfNodes, n_labels)
        self.assertGreater(lr.numberOfEdges, 100)

    def test_features_toy_data(self):
        seg, n_labels = self.generate_toy_data()
        affs = self.generate_toy_affinities()
        lr = nlr.longRangeAdjacency(seg, 2, n_labels,
                                    numberOfThreads=1)
        expected_mean = {1: 1., 2: 0.}
        for z_dir in (1, 2):
            features = nlr.longRangeFeatures(lr, seg, affs, z_dir)
            self.assertTrue((features[:, 0] == expected_mean[z_dir]).all())
            self.assertTrue((features[:, 1] == 0.).all())

    def test_features_random_data(self):
        seg, n_labels = self.generate_random_data()
        for long_range in (2, 3, 4):
            affinities = self.generate_random_affinities(long_range)
            lr = nlr.longRangeAdjacency(seg, long_range, n_labels, numberOfThreads=1)
            for z_dir in (1, 2):
                features = nlr.longRangeFeatures(lr, seg, affinities, z_dir)
                self.assertEqual(features.shape[0], lr.numberOfEdges)
                self.assertEqual(features.shape[1], 9)
                for col in range(features.shape[1]):
                    self.assertFalse(np.sum(features[:, col]) == 0)


if __name__ == '__main__':
    unittest.main()
