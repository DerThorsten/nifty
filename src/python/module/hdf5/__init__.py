from .. import Configuration

__all__ = []

if Configuration.WITH_HDF5:
    from _hdf5 import *
    for key in _hdf5.__dict__.keys():
        __all__.append(key)
else:
    pass

try:
    import h5py
    hasH5Py = True
except ImportError:
    hasH5py = False

import math



if Configuration.WITH_HDF5:


    def unionFindWatershed(heightMapFilename,
                           heightMapDataset,
                           labelsFilename,
                           labelsDataset,
                           blockShape,
                           numberOfThreads=-1):

        nDim  =  len(blockShape)
        # pow2 blockshape
        # 
        #pow2Blockshape = [2**(math.ceil(math.log(v)/math.log(2))) for v in blockShape]
        blockShapeArray = [128] * nDim

        fname = "blockwiseWatershed_float32_uint32_%dd"%nDim

        f = _hdf5.__dict__[fname]

        f(heightMapFilename, heightMapDataset, 
          labelsFilename, labelsDataset, 
          list(blockShape), list(blockShapeArray),int(numberOfThreads))


    def __extendHdf5Array():
        hdf5Arrays = [
            Hdf5ArrayUInt8,
            Hdf5ArrayUInt16,
            Hdf5ArrayUInt32,
            Hdf5ArrayUInt64,
            Hdf5ArrayInt8,
            Hdf5ArrayInt16,
            Hdf5ArrayInt32,
            Hdf5ArrayInt64,
            Hdf5ArrayFloat32,
            Hdf5ArrayFloat64
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


    __extendHdf5Array()
    del __extendHdf5Array


