from __future__ import print_function

import h5py
import numpy

import pylab
import matplotlib
import matplotlib.cm as cm
import Image
import vigra

import nifty
import nifty.ground_truth as ngt




rawF = "/home/tbeier/ilastikdata/block_73.h5"
gtF = "/home/tbeier/ilastikdata/gt_raw_c_73.h5"
growMapF = "/home/tbeier/ilastikdata/grow_map_73.h5"
gtFixedF = "/home/tbeier/ilastikdata/gt_fixed_c_73.h5"



gt  = h5py.File(gtF)['data'][:].squeeze().T


raw = h5py.File(rawF)['data'][:].squeeze()

if False:
    

    copts = vigra.blockwise.convolutionOptions([100,100,100],sigma=4.0)
    rawVigra = numpy.require(raw,requirements='F',dtype='float32')
    growMap = vigra.blockwise.hessianOfGaussianFirstEigenvalue(rawVigra, options=copts)
    copts = vigra.blockwise.convolutionOptions([100,100,100],sigma=2.0)
    growMap = vigra.blockwise.gaussianSmooth(growMap, options=copts)
    growMap = numpy.require(growMap,requirements='C',dtype='float32')

    growMapFile = h5py.File(growMapF)
    growMapFile['data'] = growMap
    growMapFile.close()

else:
    growMap  = h5py.File(growMapF)['data'][:].squeeze()





if False:
    cmVals =  numpy.random.rand ( 1000,4)
    cmVals[:,0] = 0.1
    cmVals[0,:] = 0
    randcm = matplotlib.colors.ListedColormap (cmVals)

    f = pylab.figure()
    pylab.imshow(growMap[:,:,0],cmap='gray')
    pylab.imshow( gt[:,:,0],cmap=randcm)
    pylab.show()


print('growMap',growMap.shape,growMap.dtype)
print('gt',gt.shape,gt.dtype)



if True:
    growMap-= growMap.min()
    growMap/= growMap.max()
    fixedGt = ngt.postProcessCarvingNeuroGroundTruth(growMap=growMap,groundTruth=gt,
        numberOfQueues=256,
        shrinkSizeBg=6,
        shrinkSizeObjects=10)

    print(fixedGt.min(), fixedGt.max())

    #fixedGtFile = h5py.File(gtFixedF)
    #fixedGtFile['data'] = fixedGt
    #fixedGtFile.close()
else:
    fixedGt = h5py.File(gtFixedF)['data']





segDataDict={
    'gtRaw':gt,
    'gtFixed':fixedGt,
}

rawDataDict={
    'raw':raw,
}

nifty.addHocViewer(grayData=rawDataDict, segData=segDataDict)


