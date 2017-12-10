from __future__ import print_function
import shutil
import os
import unittest
import numpy
import nifty
import nifty.graph.rag as nrag

from test_rag import TestRagBase


# TODO same test skeletons for all rag implementations
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

    def small_array_test(self, array, ragFunction):
        rag = ragFunction(array,
                          numberOfLabels=self.labels.max() + 1,
                          numberOfThreads=-1)

        self.generic_rag_test(rag=rag,
                              numberOfNodes=self.labels.max() + 1,
                              shouldEdges=self.shouldEdges,
                              shouldNotEdges=self.shouldNotEdges)

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
        self.small_array_test((self.path, "data"), nrag.gridRagStacked2DZ5)

    def big_array_test(self, array, ragFunction):
        rag = ragFunction(array,
                          numberOfLabels=self.bigLabels.max() + 1,
                          numberOfThreads=-1)

        self.generic_rag_test(rag=rag,
                              numberOfNodes=self.bigLabels.max() + 1,
                              shouldEdges=self.bigShouldEdges,
                              shouldNotEdges=self.bigShouldNotEdges)

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
    def test_grid_rag_z5_stacked2d(self):
        import z5py
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
        self.big_array_test((self.path, "data"), nrag.gridRagStacked2DZ5)


    # TODO make test skeleton for this
    @unittest.skipUnless(nifty.Configuration.WITH_HDF5, "skipping hdf5 tests")
    def test_stacked_rag_serialize_deserialize(self):
        import nifty.hdf5 as nhdf5

        shape = [3,4,4]
        chunkShape = [1,2,1]

        hidT = nhdf5.createFile(self.path)
        array = nhdf5.Hdf5ArrayUInt32(hidT, "data", shape, chunkShape)

        self.assertEqual(array.shape[0], shape[0])
        self.assertEqual(array.shape[1], shape[1])
        self.assertEqual(array.shape[2], shape[2])

        labels = [[[0, 0, 0, 0],
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
                   [4, 4, 5, 5]]]
        labels = numpy.array(labels, dtype='uint32')

        self.assertEqual(labels.shape[0], shape[0])
        self.assertEqual(labels.shape[1], shape[1])
        self.assertEqual(labels.shape[2], shape[2])

        array[0:shape[0], 0:shape[1], 0:shape[2]] = labels
        ragA = nrag.gridRagStacked2DHdf5(array,
                                         numberOfLabels=labels.max() + 1,
                                        numberOfThreads=-1)

        shouldEdges = [(0, 1),
                       (0, 3),
                       (1, 2),
                       (1, 3),
                       (2, 3),
                       (3, 4),
                       (3, 5),
                       (4, 5)]

        shouldNotEdges = [(0, 4),
                          (0, 5),
                          (1, 4),
                          (1, 5),
                          (2, 4),
                          (2, 5)]

        self.generic_rag_test(rag=ragA,
                              numberOfNodes=labels.max()+1,
                              shouldEdges=shouldEdges,
                              shouldNotEdges=shouldNotEdges)

        serialization = ragA.serialize()
        ragB = nrag.gridRagStacked2DHdf5(array,
                                         numberOfLabels=labels.max() + 1,
                                         serialization=serialization)
        self.generic_rag_test(rag=ragB,
                              numberOfNodes=labels.max()+1,
                              shouldEdges=shouldEdges, shouldNotEdges=shouldNotEdges)

        self.assertTrue((ragA.minMaxLabelPerSlice() == ragB.minMaxLabelPerSlice()).all())
        self.assertTrue((ragA.numberOfNodesPerSlice() == ragB.numberOfNodesPerSlice()).all())
        self.assertTrue((ragA.numberOfInSliceEdges() == ragB.numberOfInSliceEdges()).all())
        self.assertTrue((ragA.numberOfInBetweenSliceEdges() == ragB.numberOfInBetweenSliceEdges()).all())
        self.assertTrue((ragA.inSliceEdgeOffset() == ragB.inSliceEdgeOffset()).all())
        self.assertTrue((ragA.betweenSliceEdgeOffset() == ragB.betweenSliceEdgeOffset()).all())
        nhdf5.closeFile(hidT)


if __name__ == '__main__':
    unittest.main()
