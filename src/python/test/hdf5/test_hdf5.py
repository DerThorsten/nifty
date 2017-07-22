from __future__ import print_function
import nifty
import numpy
import os
import tempfile
import shutil

hasH5py = True
try:
    import h5py
except:
    hasH5py = False

def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)





if nifty.Configuration.WITH_HDF5:
    import nifty.hdf5
    nhdf5 = nifty.hdf5

    if hasH5py:

        def testHdf5ArrayReadFromExistingH5pyChunked():
            tempFolder = tempfile.mkdtemp()
            ensureDir(tempFolder)
            fpath = os.path.join(tempFolder,'_nifty_test_array__.h5')
            try:


                # try catch since dataset can only be created once
                shape = (101, 102, 103)
                data = numpy.ones(shape=shape, dtype='uint64')
                f = h5py.File(fpath)
                f.create_dataset("data", shape, dtype='uint64', data=data,chunks=(10,20,30))
                f.close()



                hidT = nhdf5.openFile(fpath)


                array = nhdf5.Hdf5ArrayUInt64(hidT, "data")

                assert array.ndim == 3
                shape = array.shape
                assert len(shape) == 3
                assert shape[0] == 101
                assert shape[1] == 102
                assert shape[2] == 103

                assert array.isChunked
                chunkShape = array.chunkShape
                assert len(chunkShape) == 3
                assert chunkShape[0] == 10
                assert chunkShape[1] == 20
                assert chunkShape[2] == 30

                subarray  = array[0:10,0:10 ,0:10]

            finally:
                try:
                    os.remove(fpath)
                    shutil.rmtree(tempFolder)

                except:
                    pass

        def testHdf5ArrayReadFromExistingH5pyNonChunked():
            tempFolder = tempfile.mkdtemp()
            ensureDir(tempFolder)
            fpath = os.path.join(tempFolder,'_nifty_test_array_.h5')
            try:


                # try catch since dataset can only be created once
                shape = (101, 102, 103)
                data = numpy.ones(shape=shape, dtype='uint64')
                f = h5py.File(fpath)
                f.create_dataset("data", shape, dtype='uint64', data=data)
                f.close()



                hidT = nhdf5.openFile(fpath)
                array = nhdf5.Hdf5ArrayUInt64(hidT, "data")

                assert array.ndim == 3
                shape = array.shape
                assert len(shape) == 3
                assert shape[0] == 101
                assert shape[1] == 102
                assert shape[2] == 103

                assert not array.isChunked
                chunkShape = array.chunkShape
                assert len(chunkShape) == 3
                assert chunkShape[0] == 101
                assert chunkShape[1] == 102
                assert chunkShape[2] == 103

                subarray  = array[0:10,0:10 ,0:10]
            finally:
                try:
                    os.remove(fpath)
                    shutil.rmtree(tempFolder)

                except:
                    pass

    def testHdf5ArrayCreateChunked():
        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_test_array_.h5')

        try:
            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt64(hidT, "data", [101,102,103], [11,12,13])

            assert array.ndim == 3
            shape = array.shape
            assert shape[0] == 101
            assert shape[1] == 102
            assert shape[2] == 103

            chunkShape = array.chunkShape
            assert chunkShape[0] == 11
            assert chunkShape[1] == 12
            assert chunkShape[2] == 13

            ends = [10,11,12]

            toWrite = numpy.arange(ends[0]*ends[1]*ends[2]).reshape(ends)
            array[0:ends[0], 0:ends[1], 0:ends[2]] = toWrite
            subarray  = array[0:ends[0], 0:ends[1], 0:ends[2]]

            assert numpy.array_equal(toWrite, subarray) == True

        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)
            except:
                pass


    def testHdf5ArrayCreateChunkedZipped():
        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_test_array_.h5')

        try:
            hidT = nhdf5.createFile(fpath)
            array = nhdf5.Hdf5ArrayUInt64(
                groupHandle=hidT,
                datasetName="data",
                shape=[101,102,103],
                chunkShape=[11,12,13],
                compression=9
            )

            assert array.ndim == 3
            shape = array.shape
            assert shape[0] == 101
            assert shape[1] == 102
            assert shape[2] == 103

            chunkShape = array.chunkShape
            assert chunkShape[0] == 11
            assert chunkShape[1] == 12
            assert chunkShape[2] == 13

            ends = [10,11,12]

            toWrite = numpy.arange(ends[0]*ends[1]*ends[2]).reshape(ends)
            array[0:ends[0], 0:ends[1], 0:ends[2]] = toWrite
            subarray  = array[0:ends[0], 0:ends[1], 0:ends[2]]

            assert numpy.array_equal(toWrite, subarray) == True

        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)
            except:
                pass

    def testHdf5Offsets():
        from itertools import combinations

        tempFolder = tempfile.mkdtemp()
        ensureDir(tempFolder)
        fpath = os.path.join(tempFolder,'_nifty_test_array_.h5')

        try:
            hidT = nhdf5.createFile(fpath)
            shape = [100,100,100]
            array = nhdf5.hdf5Array(
                'uint32',
                groupHandle=hidT,
                datasetName="data",
                shape=shape,
                chunkShape=[32,32,32]
            )

            testData = numpy.arange(100**3, dtype='uint32').reshape(tuple(shape))
            array.writeSubarray([0,0,0], testData)

            testOffsets = ([1,1,1], [10,0,0], [0,10,0], [0,0,10], [10,10,10])

            for offFront, offBack in combinations(testOffsets, 2):
                array.setOffsetFront(offFront)
                array.setOffsetBack(offBack)

                # check for correct effective shape
                effectiveShape = numpy.array(array.shape)
                expectedShape  = numpy.array(shape) - numpy.array(offFront) - numpy.array(offBack)
                assert numpy.array_equal(effectiveShape, expectedShape) == True

                # check for correct data
                subData = array.readSubarray([0,0,0], effectiveShape)
                assert subData.shape == tuple(effectiveShape)
                bb = numpy.s_[
                    offFront[0]:shape[0]-offBack[0],
                    offFront[1]:shape[1]-offBack[1],
                    offFront[2]:shape[2]-offBack[2]
                ]
                expectedData = testData[bb]
                assert expectedData.shape == subData.shape, "%s, %s" % (str(expectedData.shape), str(subData.shape))
                assert numpy.array_equal(subData, expectedData) == True

            # check correct handling of incorrect offsets
            for offFront in ([101,0,0],[0,101,0],[0,0,101],[101,101,101]):
                offCheck = array.setOffsetFront(offFront)
                assert not offCheck
                assert array.shape == shape

            #print("Passed!")

        finally:
            try:
                os.remove(fpath)
                shutil.rmtree(tempFolder)
            except:
                pass



#if __name__ == '__main__':
#    testHdf5Offsets()
