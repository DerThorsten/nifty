import unittest
import os
from shutil import rmtree

import numpy as np
import nifty
import nifty.graph.rag as nrag


class TestAccumulateStacked(unittest.TestCase):
    shape = (10, 256, 256)
    # shape = (3, 128, 128)

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

    def check_features(self, features, expected_length):
        self.assertEqual(len(features), expected_length)
        for feat_id in range(features.shape[1]):
            self.assertFalse(np.allclose(features[:, feat_id], 0.))

    def accumulation_in_core_test(self, accumulation_function):
        rag = nrag.gridRagStacked2D(self.labels,
                                    numberOfLabels=self.n_labels,
                                    numberOfThreads=-1)
        n_edges_xy = rag.totalNumberOfInSliceEdges
        n_edges_z  = rag.totalNumberOfInBetweenSliceEdges

        # test complete accumulation
        # print("Complete Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  numberOfThreads=-1)
        self.check_features(feats_xy, n_edges_xy)
        self.check_features(feats_z, n_edges_z)
        # print("... passed")

        # test xy-feature accumulation
        # print("Complete XY Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  keepXYOnly=True,
                                                  numberOfThreads=-1)
        self.check_features(feats_xy, n_edges_xy)
        self.assertEqual(len(feats_z), 1)
        # print("... passed")

        # test z-feature accumulation for all 3 directions
        # print("Complete Z Accumulations ...")
        for z_direction in (0, 1, 2):
            feats_xy, feats_z = accumulation_function(rag,
                                                      self.data,
                                                      keepZOnly=True,
                                                      zDirection=z_direction,
                                                      numberOfThreads=-1)
            self.assertEqual(len(feats_xy), 1)
            self.check_features(feats_z, n_edges_z)
        # print("... passed")

    def test_standard_features_in_core(self):
        self.accumulation_in_core_test(nrag.accumulateEdgeStandardFeatures)

    @unittest.skipUnless(nifty.Configuration.WITH_FASTFILTERS, "skipping fastfilter tests")
    def test_features_from_filters_in_core(self):
        self.accumulation_in_core_test(nrag.accumulateEdgeFeaturesFromFilters)

    def accumulation_z5_test(self, accumulation_function, n_feats):
        import z5py
        import nifty.z5
        label_path = os.path.join(self.tmp_dir, 'labels.n5')
        f_labels = z5py.File(label_path, use_zarr_format=False)
        dsl = f_labels.create_dataset('data',
                                      dtype='uint32',
                                      shape=self.shape,
                                      chunks=(1, 25, 25),
                                      compressor='raw')
        dsl[:] = self.labels

        rag = nrag.gridRagStacked2DZ5(nifty.z5.datasetWrapper('uint32',
                                                              os.path.join(label_path, 'data')),
                                      numberOfLabels=self.n_labels,
                                      numberOfThreads=1)
        n_edges_xy = rag.totalNumberOfInSliceEdges
        n_edges_z  = rag.totalNumberOfInBetweenSliceEdges

        data_path = os.path.join(self.tmp_dir, 'data.n5')
        f_data = z5py.File(data_path, use_zarr_format=False)
        dsd = f_data.create_dataset('data',
                                    dtype='float32',
                                    shape=self.shape,
                                    chunks=(1, 25, 25),
                                    compressor='raw')
        dsd[:] = self.data

        def open_features(keep_xy=False, keep_z=False):
            p_xy = os.path.join(self.tmp_dir, 'xy.n5')
            p_z = os.path.join(self.tmp_dir, 'z.n5')
            f_xy = z5py.File(p_xy, use_zarr_format=False)
            f_z = z5py.File(p_z, use_zarr_format=False)
            f_xy.create_dataset('data',
                                dtype='float32',
                                shape=(1 if keep_z else n_edges_xy, n_feats),
                                chunks=(1 if keep_z else 500, n_feats),
                                compressor='raw')
            f_z.create_dataset('data',
                               dtype='float32',
                               shape=(1 if keep_xy else n_edges_z, n_feats),
                               chunks=(1 if keep_xy else 500, n_feats),
                               compressor='raw')
            return p_xy, p_z

        def load_features(p_xy, p_z):
            xy_feats = z5py.File(p_xy)['data'][:]
            z_feats = z5py.File(p_z)['data'][:]
            rmtree(p_xy)
            rmtree(p_z)
            return xy_feats, z_feats

        path_xy, path_z = open_features()

        # test complete accumulation
        print("Complete Accumulation ...")
        accumulation_function(rag,
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(data_path, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_xy, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_z, 'data')),
                              numberOfThreads=1)

        feats_xy, feats_z = load_features(path_xy, path_z)
        self.check_features(feats_xy, n_edges_xy)
        self.check_features(feats_z, n_edges_z)
        print("... passed")

        # test xy-feature accumulation
        print("Complete XY Accumulation ...")
        path_xy, path_z = open_features(keep_xy=True)
        accumulation_function(rag,
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(data_path, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_xy, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_z, 'data')),
                              keepXYOnly=True,
                              numberOfThreads=-1)

        feats_xy, feats_z = load_features(path_xy, path_z)

        self.check_features(feats_xy, n_edges_xy)
        self.assertEqual(len(feats_z), 1)
        print("... passed")

        # test z-feature accumulation for all 3 directions
        print("Complete Z Accumulations ...")
        for z_direction in (0, 1, 2):
            path_xy, path_z = open_features(keep_z=True)
            accumulation_function(rag,
                                  nifty.z5.datasetWrapper('float32',
                                                          os.path.join(data_path, 'data')),
                                  nifty.z5.datasetWrapper('float32',
                                                          os.path.join(path_xy, 'data')),
                                  nifty.z5.datasetWrapper('float32',
                                                          os.path.join(path_z, 'data')),
                                  keepZOnly=True,
                                  numberOfThreads=-1)

            feats_xy, feats_z = load_features(path_xy, path_z)
            self.assertEqual(len(feats_xy), 1)
            self.check_features(feats_z, n_edges_z)
        print("... passed")

    @unittest.skipUnless(nifty.Configuration.WITH_Z5, "skipping z5 tests")
    def test_z5_standard_features(self):
        self.accumulation_z5_test(nrag.accumulateEdgeStandardFeatures, n_feats=9)

    @unittest.skipUnless(nifty.Configuration.WITH_Z5 and nifty.Configuration.WITH_FASTFILTERS,
                         "skipping z5 fastfilter tests")
    def test_z5_features_from_filters(self):
        self.accumulation_z5_test(nrag.accumulateEdgeFeaturesFromFilters, n_feats=9 * 12)

    @unittest.skipUnless(nifty.Configuration.WITH_Z5, "skipping z5 tests")
    def test_in_vs_out_of_core(self):
        accumulation_function = nrag.accumulateEdgeStandardFeatures
        n_feats = 9
        import z5py
        import nifty.z5

        ###############
        # get features with out of core calculation

        label_path = os.path.join(self.tmp_dir, 'labels.n5')
        f_labels = z5py.File(label_path, use_zarr_format=False)
        dsl = f_labels.create_dataset('data',
                                      dtype='uint32',
                                      shape=self.shape,
                                      chunks=(1, 25, 25),
                                      compressor='raw')
        dsl[:] = self.labels

        rag_ooc = nrag.gridRagStacked2DZ5(nifty.z5.datasetWrapper('uint32',
                                                                  os.path.join(label_path,
                                                                               'data')),
                                          numberOfLabels=self.n_labels,
                                          numberOfThreads=1)
        data_path = os.path.join(self.tmp_dir, 'data.n5')
        f_data = z5py.File(data_path, use_zarr_format=False)
        dsd = f_data.create_dataset('data',
                                    dtype='float32',
                                    shape=self.shape,
                                    chunks=(1, 25, 25),
                                    compressor='raw')
        dsd[:] = self.data

        n_edges_xy = rag_ooc.totalNumberOfInSliceEdges
        n_edges_z  = rag_ooc.totalNumberOfInBetweenSliceEdges

        def open_features(keep_xy=False, keep_z=False):
            p_xy = os.path.join(self.tmp_dir, 'xy.n5')
            p_z = os.path.join(self.tmp_dir, 'z.n5')
            f_xy = z5py.File(p_xy, use_zarr_format=False)
            f_z = z5py.File(p_z, use_zarr_format=False)
            f_xy.create_dataset('data',
                                dtype='float32',
                                shape=(1 if keep_z else n_edges_xy, n_feats),
                                chunks=(1 if keep_z else 500, n_feats),
                                compressor='raw')
            f_z.create_dataset('data',
                               dtype='float32',
                               shape=(1 if keep_xy else n_edges_z, n_feats),
                               chunks=(1 if keep_xy else 500, n_feats),
                               compressor='raw')
            return p_xy, p_z

        def load_features(p_xy, p_z):
            xy_feats = z5py.File(p_xy)['data'][:]
            z_feats = z5py.File(p_z)['data'][:]
            rmtree(p_xy)
            rmtree(p_z)
            return xy_feats, z_feats

        path_xy, path_z = open_features()

        # test complete accumulation
        accumulation_function(rag_ooc,
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(data_path, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_xy, 'data')),
                              nifty.z5.datasetWrapper('float32',
                                                      os.path.join(path_z, 'data')),
                              numberOfThreads=-11)

        feats_xy_ooc, feats_z_ooc = load_features(path_xy, path_z)

        ###############
        # get features with in core calculation

        rag = nrag.gridRagStacked2D(self.labels,
                                    numberOfLabels=self.n_labels,
                                    numberOfThreads=1)

        # test complete accumulation
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  numberOfThreads=1)
        self.assertEqual(feats_xy.shape, feats_xy_ooc.shape)
        self.assertTrue(np.allclose(feats_xy, feats_xy_ooc))
        self.assertEqual(feats_z.shape, feats_z_ooc.shape)
        self.assertTrue(np.allclose(feats_z, feats_z_ooc))

    @staticmethod
    def make_labels_with_ignore(shape):
        labels = np.zeros(shape, dtype='uint32')
        mask = np.random.choice([0, 1], size=shape, p=[1. / 10., 9. / 10.]).astype('bool')
        label = 0
        for z in range(shape[0]):
            for y in range(shape[1]):
                for x in range(shape[2]):
                    if not mask[z, y, x]:
                        continue
                    labels[z, y, x] = label
                    if np.random.random() > .95:
                        have_increased = True
                        label += 1
                    else:
                        have_increased = False
            if not have_increased:
                label += 1
        return labels

    def ignore_label_test_test(self, accumulation_function):
        labels_with_ignore = self.make_labels_with_ignore(self.shape)
        rag = nrag.gridRagStacked2D(labels_with_ignore,
                                    numberOfLabels=labels_with_ignore.max() + 1,
                                    ignoreLabel=0,
                                    numberOfThreads=1)
        n_edges_xy = rag.totalNumberOfInSliceEdges
        n_edges_z  = rag.totalNumberOfInBetweenSliceEdges

        # test complete accumulation
        print("Complete Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  numberOfThreads=1)
        self.check_features(feats_xy, n_edges_xy)
        self.check_features(feats_z, n_edges_z)
        print("... passed")

        # test xy-feature accumulation
        print("Complete XY Accumulation ...")
        feats_xy, feats_z = accumulation_function(rag,
                                                  self.data,
                                                  keepXYOnly=True,
                                                  numberOfThreads=-1)
        self.check_features(feats_xy, n_edges_xy)
        self.assertEqual(len(feats_z), 1)
        print("... passed")

        # test z-feature accumulation for all 3 directions
        print("Complete Z Accumulations ...")
        for z_direction in (0, 1, 2):
            feats_xy, feats_z = accumulation_function(rag,
                                                      self.data,
                                                      keepZOnly=True,
                                                      zDirection=z_direction,
                                                      numberOfThreads=-1)
            self.assertEqual(len(feats_xy), 1)
            self.check_features(feats_z, n_edges_z)
        print("... passed")

    def test_standard_features_ignore(self):
        self.ignore_label_test_test(nrag.accumulateEdgeStandardFeatures)


if __name__ == '__main__':
    unittest.main()
