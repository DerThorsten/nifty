from __future__ import print_function
import os
import unittest
import numpy
import nifty
import nifty.graph.rag as nrag

try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

from test_rag import TestRagBase


class TestRagStacked(TestRagBase):
    # shape of the small label array
    shape = [3, 2, 2]
    # small labels
    labels = numpy.array([[[0, 1],
                           [0, 1]],
                          [[2, 2],
                           [2, 3]],
                          [[4, 5],
                           [6, 6]]],
                         dtype='uint32')
    # graph edges for small labels
    shouldEdges = [(0, 1),
                   (0, 2),
                   (1, 2),
                   (1, 3),
                   (2, 3),
                   (2, 4),
                   (2, 5),
                   (2, 6),
                   (3, 6),
                   (4, 5),
                   (4, 6),
                   (5, 6)]
    # not graph edges for small labels
    shouldNotEdges = [(0, 3),
                      (0, 4),
                      (0, 5),
                      (0, 6),
                      (1, 4),
                      (1, 5),
                      (1, 6)]

    # shape of the big label array
    bigShape = [3, 4, 4]
    # big labels
    bigLabels = numpy.array([[[0, 0, 0, 0],
                              [1, 1, 1, 1],
                              [2, 2, 2, 2],
                              [2, 2, 2, 2]],
                             [[3, 3, 3, 3],
                              [3, 3, 3, 3],
                              [3, 3, 3, 3],
                              [3, 3, 3, 3]],
                             [[4, 4, 5, 5],
                              [4, 4, 5, 5],
                              [4, 4, 5, 5],
                              [4, 4, 5, 5]]],
                            dtype='uint32')
    # graph edges for big labels
    bigShouldEdges = [(0, 1),
                      (0, 3),
                      (1, 2),
                      (1, 3),
                      (2, 3),
                      (3, 4),
                      (3, 5),
                      (4, 5)]
    # not graph edges for big labels
    bigShouldNotEdges = [(0, 4),
                         (0, 5),
                         (1, 4),
                         (1, 5),
                         (2, 4),
                         (2, 5)]
    # graph edges for ignore graphs
    ignoreShouldEdges = [(1, 2),
                         (1, 3),
                         (2, 3),
                         (3, 4),
                         (3, 5),
                         (4, 5)]
    # not graph edges for ignore graphs
    ignoreShouldNotEdges = [(0, 1),
                            (0, 3),
                            (0, 4),
                            (0, 5),
                            (1, 4),
                            (1, 5),
                            (2, 4),
                            (2, 5)]

    def ignore_array_test(self, array, ragFunction):
        rag = ragFunction(array,
                          numberOfLabels=self.bigLabels.max() + 1,
                          ignoreLabel=0,
                          numberOfThreads=1)

        self.generic_rag_test(rag=rag,
                              numberOfNodes=self.bigLabels.max() + 1,
                              shouldEdges=self.ignoreShouldEdges,
                              shouldNotEdges=self.ignoreShouldNotEdges,
                              shape=self.bigShape)

    def test_grid_rag_stacked_ignore(self):
        array = numpy.zeros(self.bigShape, dtype='uint32')
        array[:] = self.bigLabels
        self.ignore_array_test(array, nrag.gridRagStacked2D)

    def small_array_test(self, array, ragFunction):
        rag = ragFunction(array,
                          numberOfLabels=self.labels.max() + 1,
                          numberOfThreads=-1)

        self.generic_rag_test(rag=rag,
                              numberOfNodes=self.labels.max() + 1,
                              shouldEdges=self.shouldEdges,
                              shouldNotEdges=self.shouldNotEdges,
                              shape=self.shape)

    def test_grid_rag_stacked2d(self):
        array = numpy.zeros(self.shape, dtype='uint32')
        array[:] = self.labels
        self.small_array_test(array, nrag.gridRagStacked2D)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_grid_rag_hdf5_stacked2d(self):
        import nifty.hdf5 as nhdf5
        hidT = nhdf5.createFile(self.path)
        chunkShape = [1, 2, 1]
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", self.shape, chunkShape)
        array[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]] = self.labels
        self.small_array_test(array, nrag.gridRagStacked2DHdf5)
        nhdf5.closeFile(hidT)

    @unittest.skipUnless(nifty.Configuration.WITH_Z5, "skipping z5 tests")
    def test_grid_rag_z5_stacked2d(self):
        import z5py
        import nifty.z5
        zfile = z5py.File(self.path, use_zarr_format=False)
        chunkShape = [1, 2, 1]
        array = zfile.create_dataset("data",
                                     dtype='uint32',
                                     shape=self.shape,
                                     chunks=chunkShape,
                                     compressor='raw')
        array[:] = self.labels
        # we only pass the path and key to the dataset, because we
        # cannot properly link the python bindings for now
        # self.small_array_test(array, nrag.gridRagStacked2DZ5)
        self.small_array_test(nifty.z5.datasetWrapper('uint32', os.path.join(self.path, 'data')),
                              nrag.gridRagStacked2DZ5)

    def big_array_test(self, array, ragFunction):
        rag = ragFunction(array,
                          numberOfLabels=self.bigLabels.max() + 1,
                          numberOfThreads=-1)

        self.generic_rag_test(rag=rag,
                              numberOfNodes=self.bigLabels.max() + 1,
                              shouldEdges=self.bigShouldEdges,
                              shouldNotEdges=self.bigShouldNotEdges,
                              shape=self.bigShape)

    def test_grid_rag_stacked2d_large(self):
        array = numpy.zeros(self.bigShape, dtype='uint32')
        array[:] = self.bigLabels
        self.big_array_test(array, nrag.gridRagStacked2D)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_grid_rag_hdf5_stacked2d_large(self):
        import nifty.hdf5 as nhdf5
        hidT = nhdf5.createFile(self.path)
        chunkShape = [1, 2, 1]
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", self.bigShape, chunkShape)
        array[0:self.bigShape[0], 0:self.bigShape[1], 0:self.bigShape[2]] = self.bigLabels
        self.big_array_test(array, nrag.gridRagStacked2DHdf5)
        nhdf5.closeFile(hidT)

    @unittest.skipUnless(nifty.Configuration.WITH_Z5, "skipping z5 tests")
    def test_grid_rag_z5_stacked2d_large(self):
        import z5py
        import nifty.z5
        zfile = z5py.File(self.path, use_zarr_format=False)
        chunkShape = [1, 2, 1]
        array = zfile.create_dataset("data",
                                     dtype='uint32',
                                     shape=self.bigShape,
                                     chunks=chunkShape,
                                     compressor='raw')
        array[:] = self.bigLabels
        # we only pass the path and key to the dataset, because we
        # cannot properly link the python bindings for now
        # self.small_array_test(array, nrag.gridRagStacked2DZ5)
        self.big_array_test(nifty.z5.datasetWrapper('uint32', os.path.join(self.path, 'data')),
                            nrag.gridRagStacked2DZ5)

    def serialization_test(self, array, ragFunction):
        ragA = ragFunction(array,
                           numberOfLabels=self.bigLabels.max() + 1,
                           numberOfThreads=-1)

        self.generic_rag_test(rag=ragA,
                              numberOfNodes=self.bigLabels.max() + 1,
                              shouldEdges=self.bigShouldEdges,
                              shouldNotEdges=self.bigShouldNotEdges,
                              shape=self.bigShape)
        nrag.writeStackedRagToHdf5(ragA, self.path2)
        ragB = nrag.readStackedRagFromHdf5(labels=array,
                                           numberOfLabels=self.bigLabels.max() + 1,
                                           savePath=self.path2)
        self.generic_rag_test(rag=ragB,
                              numberOfNodes=self.bigLabels.max() + 1,
                              shouldEdges=self.bigShouldEdges,
                              shouldNotEdges=self.bigShouldNotEdges,
                              shape=self.bigShape)

        self.assertEqual(type(ragA), type(ragB))

        self.assertTrue((ragA.minMaxLabelPerSlice() == ragB.minMaxLabelPerSlice()).all())
        self.assertTrue((ragA.numberOfNodesPerSlice() == ragB.numberOfNodesPerSlice()).all())
        self.assertTrue((ragA.numberOfInSliceEdges() == ragB.numberOfInSliceEdges()).all())
        self.assertTrue((ragA.numberOfInBetweenSliceEdges() == ragB.numberOfInBetweenSliceEdges()).all())
        self.assertTrue((ragA.inSliceEdgeOffset() == ragB.inSliceEdgeOffset()).all())
        self.assertTrue((ragA.betweenSliceEdgeOffset() == ragB.betweenSliceEdgeOffset()).all())

    @unittest.skipUnless(WITH_H5PY, "skipping explicit serialization tests")
    def test_stacked_rag_serialize_deserialize(self):
        array = numpy.zeros(self.bigShape, dtype='uint32')
        array[:] = self.bigLabels
        self.serialization_test(array, nrag.gridRagStacked2D)

    @unittest.skipUnless(nifty.Configuration.WITH_HDF5 and WITH_H5PY,
                         "skipping hdf5 serialization tests")
    def test_stacked_rag_hdf5_serialize_deserialize(self):
        import nifty.hdf5 as nhdf5
        hidT = nhdf5.createFile(self.path)
        chunkShape = [1, 2, 1]
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", self.bigShape, chunkShape)
        array[0:self.bigShape[0], 0:self.bigShape[1], 0:self.bigShape[2]] = self.bigLabels
        serialization_test(self, array, gridRagStacked2DHdf5)
        nhdf5.closeFile(hidT)

    @unittest.skipUnless(nifty.Configuration.WITH_Z5 and WITH_H5PY,
                         "skipping z5 serialization tests")
    def test_grid_rag_z5_serialize_deserialize(self):
        import z5py
        import nifty.z5
        zfile = z5py.File(self.path, use_zarr_format=False)
        chunkShape = [1, 2, 1]
        array = zfile.create_dataset("data",
                                     dtype='uint32',
                                     shape=self.bigShape,
                                     chunks=chunkShape,
                                     compressor='raw')
        array[:] = self.bigLabels
        # we only pass the path and key to the dataset, because we
        # cannot properly link the python bindings for now
        # self.small_array_test(array, nrag.gridRagStacked2DZ5)
        self.serialization_test(nifty.z5.datasetWrapper('uint32', os.path.join(self.path, "data")),
                                nrag.gridRagStacked2DZ5)


if __name__ == '__main__':
    unittest.main()
