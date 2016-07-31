from __future__ import print_function
import nifty
import numpy


hasH5py = True
try:
    import h5py
except:
    hasH5py = False


if nifty.Configuration.WITH_HDF5:

    nhdf5 = nifty.hdf5


    if hasH5py:

        def testHdf5ArrayReadFromExisting():

            # try catch since dataset can only be created once
            try:

                shape = (101, 102, 103)
                data = numpy.ones(shape=shape, dtype='uint64')

                f = h5py.File("/home/tbeier/tmp4.h5")
                f.create_dataset("data", shape, dtype='uint64', data=data,chunks=(10,10,10))
                f.close()
            except:
                pass



            hidT = nhdf5.openFile("/home/tbeier/tmp4.h5")



            array = nhdf5.Hdf5ArrayUInt64(hidT, "data")

            assert array.ndim == 3

            shape = array.shape
            assert shape[0] == 101
            assert shape[1] == 102
            assert shape[2] == 103


            subarray  = array[0:10,0:10,0:10]




    def testHdf5ArrayCreateChunked():

        hidT = nhdf5.createFile("/home/tbeier/tmp9.h5")

        array = nhdf5.Hdf5ArrayUInt64(hidT, "data", [101,102,103], [10,10,10])

        assert array.ndim == 3
        shape = array.shape
        assert shape[0] == 101
        assert shape[1] == 102
        assert shape[2] == 103

        subarray  = array[0:10,0:10,0:10]
    
