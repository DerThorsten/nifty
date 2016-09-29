from __future__ import print_function

from _nifty import *
# __all__ = []
# for key in _nifty.__dict__.keys():
    # __all__.append(key)

import types
from functools import partial
import numpy
import time
import sys

import graph



    
class Timer:
    def __init__(self, name=None, verbose=True):
        """
        @brief      Class for timer.
        """ 
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.dt = self.end - self.start
        if self.verbose:
            if self.name is not None:
                print(self.name,"took",self.dt,"sec") 
            else:
                print("took",self.dt,"sec")






if Configuration.WITH_HDF5:

    def __extendHdf5():
        hdf5Arrays = [
            hdf5.Hdf5ArrayUInt8,
            hdf5.Hdf5ArrayUInt16,
            hdf5.Hdf5ArrayUInt32,
            hdf5.Hdf5ArrayUInt64,
            hdf5.Hdf5ArrayInt8,
            hdf5.Hdf5ArrayInt16,
            hdf5.Hdf5ArrayInt32,
            hdf5.Hdf5ArrayInt64,
            hdf5.Hdf5ArrayFloat32,
            hdf5.Hdf5ArrayFloat64
        ]

        def getItem(self, slicing):
            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.readSubarray(roiBegin, roiEnd)

        def setItem(self, slicing, value):
            asArray = numpy.require(value)

            dim = self.ndim
            roiBegin = [None]*dim
            roiEnd = [None]*dim
            shape = [None]*dim
            for d in range(dim):
                sliceObj = slicing[d]
                roiBegin[d] = int(sliceObj.start)
                roiEnd[d] = int(sliceObj.stop)
                if roiEnd[d] - roiBegin[d] != asArray.shape[d]:
                    raise RuntimeError("array to write does not match slicing shape")
                step = sliceObj.step
                if step is not None and  step != 1:
                    raise RuntimeError("currently step must be 1 in slicing but step is %d"%sliceObj.step)

            return self.writeSubarray(roiBegin, asArray)


        for array in hdf5Arrays:
            array.__getitem__ = getItem
            array.__setitem__ = setItem






    __extendHdf5()
    del __extendHdf5
