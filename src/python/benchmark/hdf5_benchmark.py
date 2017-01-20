from __future__ import print_function

import nifty
import nifty.hdf5
import h5py
import threading

fileName = "/home/tbeier/Desktop/ufd_overseg6.h5"
#fileName = "/home/tbeier/_4nm_ufd_overseg.h5"
dsetName = "data"

shape = [2000,2000,2000]
blockShape = [100,100,100]



blocking = nifty.tools.blocking(roiBegin=[0]*3, roiEnd=shape, 
                                blockShape=blockShape)

numberOfBlocks = blocking.numberOfBlocks

nThreads = 40



if True:
    # now the same in python
    h5File = h5py.File(fileName,'r')
    array = h5File[dsetName]
    lock = threading.Lock()

    with nifty.Timer("python"):
        val = long(0)
        def f(blockIndex):
            global val
            block = blocking.getBlock(blockIndex)
            b,e = block.begin, block.end

            with lock:
                subarray = array[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
                _val = subarray[0,0,0]
                val += long(_val)

        nifty.tools.parallelForEach(range(numberOfBlocks),f=f, nWorkers=nThreads)
        print("val",val)

print()

if True:
    # first c++
    h5File = nifty.hdf5.openFile(fileName)
    array = nifty.hdf5.Hdf5ArrayUInt32(h5File, dsetName)

    with nifty.Timer("c++"):
        nifty.hdf5.runBenchmark(array, blocking, nThreads)

print()

if True:
    # first c++
    h5File = nifty.hdf5.openFile(fileName)
    array = nifty.hdf5.Hdf5ArrayUInt32(h5File, dsetName)
    lock = threading.Lock()
    with nifty.Timer("c++ python"):
        val = long(0)
        def f(blockIndex):
            global val
            block = blocking.getBlock(blockIndex)
            b,e = block.begin, block.end

            with lock:
                subarray = array[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
                _val = subarray[0,0,0]
                val += long(_val)

        nifty.tools.parallelForEach(range(numberOfBlocks),f=f, nWorkers=nThreads)
        print("val",val)