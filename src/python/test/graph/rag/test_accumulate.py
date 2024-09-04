import platform
import unittest

import numpy as np
import nifty.graph.rag as nrag


# Just a quick and dirty test to check that the functions can be called
class TestAccumulate(unittest.TestCase):
    shape_2d = 2 * (128,)
    shape_3d = 3 * (64,)

    def test_accumulate_mean_and_length_2d(self):
        labels = np.random.randint(0, 100, size=self.shape_2d, dtype='uint32')
        rag = nrag.gridRag(labels, numberOfLabels=100)
        data = np.random.random_sample(self.shape_2d).astype('float32')
        res = nrag.accumulateEdgeMeanAndLength(rag, data)
        self.assertTrue(np.sum(res) != 0)

    def test_accumulate_mean_and_length_3d(self):
        labels = np.random.randint(0, 100, size=self.shape_3d, dtype='uint32')
        rag = nrag.gridRag(labels, numberOfLabels=100)
        data = np.random.random_sample(self.shape_3d).astype('float32')
        res = nrag.accumulateEdgeMeanAndLength(rag, data)
        self.assertTrue(np.sum(res) != 0)

    @unittest.skipIf(platform.system() == "Darwin", "Test fails on Mac")
    def test_accumulate_affinities(self):
        labels = np.random.randint(0, 100, size=self.shape_2d, dtype='uint32')
        rag = nrag.gridRag(labels, numberOfLabels=100)
        offsets = [[-1, 0], [0, -1], [-3, 0], [0, -3]]
        aff_shape = (len(offsets),) + labels.shape
        data = np.random.random_sample(aff_shape).astype('float32')
        res = nrag.accumulateAffinityStandartFeatures(rag, data, offsets, 0.0, 1.0)
        self.assertTrue(np.sum(res) != 0)


if __name__ == '__main__':
    unittest.main()
