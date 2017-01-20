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


