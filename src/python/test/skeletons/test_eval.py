import os
import unittest
from shutil import rmtree

import numpy as np
import nifty
WITH_Z5 = nifty.Configuration.WITH_Z5

try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False


class TestSkeletons(unittest.TestCase):

    def setUp(self):
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

        # make segmenation data
        seg = np.zeros((100, 100, 100), dtype='uint64')
        seg[:50] = 1
        seg[50:, 50:, 50:] = 2
        seg[:50, :50, :50] = 3
        f_seg = z5py.File('./tmp/seg.n5', use_zarr_format=False)
        ds = f_seg.create_dataset('seg', shape=seg.shape, chunks=(20, 20, 20), dtype=seg.dtype)
        ds[:] = seg

        # make skeletons
        f_skels = z5py.File('./tmp/skels.n5', use_zarr_format=False)
        # skeleton 1: only in label 3
        skel1 = np.array([[0, 0, 0, 0],
                          [1, 10, 10, 10],
                          [2, 15, 15, 15],
                          [3, 20, 20, 20],
                          [4, 25, 25, 25],
                          [5, 35, 35, 35]], dtype='uint64')
        g1 = f_skels.create_group('1')
        c1 = g1.create_dataset('coordinates', shape=skel1.shape,
                               chunks=skel1.shape, dtype='uint64')
        c1[:] = skel1
        # skeleton 2: in label 0 and 2
        skel2 = np.array([[0, 60, 0, 0],
                          [1, 70, 10, 10],
                          [2, 75, 25, 25],
                          [3, 75, 45, 45],
                          [4, 80, 55, 55],
                          [5, 85, 65, 65]], dtype='uint64')
        g2 = f_skels.create_group('2')
        c2 = g2.create_dataset('coordinates', shape=skel2.shape,
                               chunks=skel2.shape, dtype='uint64')
        c2[:] = skel2

    def tearDown(self):
        if os.path.exists('./tmp'):
            rmtree('./tmp')

    @unittest.skipUnless(WITH_Z5 and WITH_Z5PY,
                         "Need z5 support")
    def test_nodes(self):
        import nifty.skeletons as nskel
        out = nskel.getSkeletonNodeAssignments('./tmp/seg.n5/seg', './tmp/skels.n5', [1, 2], 1)
        self.assertEqual(list(out.keys()), [1, 2])
        out1 = [out[1][k] for k in sorted(out[1].keys())]
        out2 = [out[2][k] for k in sorted(out[2].keys())]
        self.assertEqual(list(out1), 6 * [3])
        self.assertEqual(list(out2), [0, 0, 0, 0, 2, 2])


if __name__ == '__main__':
    unittest.main()
