import unittest
import numpy as np
import nifty.graph.rag as nrag



class TestCarving(unittest.TestCase):

    def make_labels(self, ndim):
        shape = ndim * (100,)
        x = np.random.randint(0, 100, size=shape,
                              dtype='uint32')
        return x

    def _test_carving(self, ndim):
        from nifty.carving import carvingSegmenter
        x = self.make_labels(ndim)
        rag = nrag.gridRag(x, numberOfLabels=int(x.max()) + 1)
        edgeWeights = 128 * np.random.rand(rag.numberOfEdges).astype('float32')
        segmenter = carvingSegmenter(rag, edgeWeights)

        noBoundaryBelow = 64.
        for bias in (.9, .95, 1.):
            seeds = np.random.randint(0, 3, size=rag.numberOfNodes)
            out = segmenter(seeds, bias, noBoundaryBelow)
            self.assertEqual(len(out), rag.numberOfNodes)
            self.assertTrue(np.allclose(np.unique(out), [1, 2]))

    def test_carving_2d(self):
        self._test_carving(2)

    def test_carving_3d(self):
        self._test_carving(3)


if __name__ == '__main__':
    unittest.main()
