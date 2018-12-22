from __future__ import print_function

import os
import unittest
from shutil import rmtree

import numpy
import nifty
WITH_HDF5 = nifty.Configuration.WITH_HDF5

try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY= False


class TestHDF5(unittest.TestCase):
    tempFolder = './tmp_hdf5'

    def setUp(self):
        try:
            os.mkdir(self.tempFolder)
        except OSError:
            pass

    def tearDown(self):
        try:
            rmtree(self.tempFolder)
        except OSError:
            pass

    @unittest.skipUnless(WITH_HDF5 and WITH_H5PY,
                         "Need nifty-hdf5 and h5py")
    def test_hdf5_read_from_chunked(self):
        import nifty.hdf5 as nhdf5
        fpath = os.path.join(self.tempFolder, '_nifty_test_array__.h5')

        shape = (101, 102, 103)
        chunks = (10, 20, 30)
        data = numpy.ones(shape=shape, dtype='uint64')
        with h5py.File(fpath) as f:
            f.create_dataset("data", shape, dtype='uint64', data=data, chunks=chunks)

        hidT = nhdf5.openFile(fpath)
        array = nhdf5.Hdf5ArrayUInt64(hidT, "data")
        ashape = array.shape

        self.assertEqual(array.ndim, 3)
        self.assertEqual(len(ashape), 3)
        self.assertEqual(tuple(ashape), shape)

        self.assertTrue(array.isChunked)
        chunkShape = array.chunkShape
        self.assertEqual(len(chunkShape), 3)
        self.assertEqual(tuple(chunkShape), tuple(chunks))

        subarray = array[0:10, 0:10, 0:10]
        expected = data[0:10, 0:10, 0:10]
        self.assertEqual(subarray.shape, expected.shape)
        self.assertTrue(numpy.allclose(subarray, expected))

    @unittest.skipUnless(WITH_HDF5 and WITH_H5PY,
                         "Need nifty-hdf5 and h5py")
    def test_hdf5_read_from_non_chunked(self):
        import nifty.hdf5 as nhdf5
        fpath = os.path.join(self.tempFolder, '_nifty_test_array_.h5')

        shape = (101, 102, 103)
        data = numpy.ones(shape=shape, dtype='uint64')
        with h5py.File(fpath) as f:
            f.create_dataset("data", shape, dtype='uint64', data=data)

        hidT = nhdf5.openFile(fpath)
        array = nhdf5.Hdf5ArrayUInt64(hidT, "data")

        ashape = array.shape
        self.assertEqual(array.ndim, 3)
        self.assertEqual(len(ashape), 3)
        self.assertEqual(shape, tuple(ashape))

        self.assertFalse(array.isChunked)
        chunkShape = array.chunkShape
        self.assertEqual(len(chunkShape), 3)
        self.assertEqual(tuple(chunkShape), shape)

        subarray = array[0:10, 0:10, 0:10]
        expected = data[0:10, 0:10, 0:10]
        self.assertEqual(subarray.shape, expected.shape)
        self.assertTrue(numpy.allclose(subarray, expected))

    @unittest.skipUnless(WITH_HDF5, "Need nifty-hdf5")
    def test_create_chunked_array(self):
        import nifty.hdf5 as nhdf5
        fpath = os.path.join(self.tempFolder, '_nifty_test_array_.h5')

        hidT = nhdf5.createFile(fpath)
        shape = [101, 102, 103]
        chunks = [11, 12, 13]
        array = nhdf5.Hdf5ArrayUInt64(hidT, "data", shape, chunks)

        ashape = array.shape
        self.assertEqual(array.ndim, 3)
        self.assertEqual(ashape, shape)

        chunkShape = array.chunkShape
        self.assertEqual(chunkShape, chunks)

        ends = [10, 11, 12]

        toWrite = numpy.arange(ends[0]*ends[1]*ends[2]).reshape(ends)
        array[0:ends[0], 0:ends[1], 0:ends[2]] = toWrite
        subarray  = array[0:ends[0], 0:ends[1], 0:ends[2]]

        self.assertTrue(numpy.array_equal(toWrite, subarray))

    @unittest.skipUnless(WITH_HDF5, "Need nifty-hdf5")
    def test_create_zipped_array(self):
        import nifty.hdf5 as nhdf5
        fpath = os.path.join(self.tempFolder, '_nifty_test_array_.h5')

        shape = [101,102,103]
        chunks = [11,12,13]
        hidT = nhdf5.createFile(fpath)
        array = nhdf5.Hdf5ArrayUInt64(
            groupHandle=hidT,
            datasetName="data",
            shape=shape,
            chunkShape=chunks,
            compression=9
        )

        ashape = array.shape
        self.assertEqual(array.ndim, 3)
        self.assertEqual(shape, ashape)

        chunkShape = array.chunkShape
        self.assertEqual(chunkShape, chunks)

        ends = [10,11,12]

        toWrite = numpy.arange(ends[0]*ends[1]*ends[2]).reshape(ends)
        array[0:ends[0], 0:ends[1], 0:ends[2]] = toWrite
        subarray = array[0:ends[0], 0:ends[1], 0:ends[2]]

        self.assertTrue(numpy.array_equal(toWrite, subarray))


if __name__ == '__main__':
    unittest.main()
