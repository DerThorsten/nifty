import nifty
import nifty.viewer
import numpy
import vigra
import h5py 

import nifty.tools
from multiprocessing import cpu_count
import pylab
import scipy.ndimage
import math
import threading
import fastfilters

pmapPath = "/home/tbeier/Desktop/hess-2nm-subsampled-autocontext-predictions.h5"
rawFile = None


pmapH5 = h5py.File(pmapPath)
pmapDset = pmapH5['predictions']





def makeBall(r):
    size = 2*r + 1

    mask = numpy.zeros([size]*3)

    for x0 in range(-1*r, r + 1):
        for x1 in range(-1*r, r + 1):
            for x2 in range(-1*r, r + 1):
                
                if math.sqrt(x0**2 + x1**2 + x2**2) <= r:
                    mask[x0+r, x1+r, x2+r] = 1

    return mask, (r,r,r)


def membraneOverseg3D(pmapDset, **kwargs):



    axisResolution = kwargs.get("axisResolution",['4nm']*3)
    featureBlockShape = kwargs.get("featureBlockShape",['100']*3)
    shape = pmapDset.shape[0:3]
    

    roiBegin = kwargs.get("roiBegin", [0]*3)
    roiEnd = kwargs.get("roiEnd", shape)
    nWorkers = kwargs.get("nWorkers",cpu_count())
    blocking = nifty.tools.blocking(roiBegin=roiBegin, roiEnd=roiEnd, blockShape=featureBlockShape)
    margin = [37 ,37,37]



    def pmapToHeightMap(pmap):
        
        footprint, origin = makeBall(r=3)

        medianImg = scipy.ndimage.percentile_filter(input=pmap, 
                                                    #size=(20,20,20),
                                                    footprint=footprint, 
                                                    #origin=origin, 
                                                    mode='reflect',
                                                    percentile=50.0)
        if False:
            blurredSmall = vigra.gaussianSmoothing(pmap.T, 1.0,).T
            blurredLarge = vigra.gaussianSmoothing(pmap.T, 6.0,).T
            blurredSuperLarge = vigra.gaussianSmoothing(pmap.T, 10.0,).T

        else:
            blurredSmall = fastfilters.gaussianSmoothing(pmap, 1.0,)
            blurredLarge = fastfilters.gaussianSmoothing(pmap, 6.0,)
            blurredSuperLarge = fastfilters.gaussianSmoothing(pmap, 10.0,)

        #combined = medianImg + blurredSuperLarge*0.3 + 0.15*blurredLarge + 0.1*blurredSmall
        #if False:
        #    nifty.viewer.view3D(pmap, show=False, title='pm',cmap='jet')
        #    nifty.viewer.view3D(medianImg, show=False, title='medianImg',cmap='jet')
        #    nifty.viewer.view3D(combined, show=False, title='combined',cmap='jet')
        #    pylab.show()
        #    print vigraArray.shape, vigraArray.strides

        return medianImg


    numberOfBlocks = blocking.numberOfBlocks
    lock = threading.Lock()
    done = [0]

    with nifty.tools.progressBar(size=numberOfBlocks) as bar:

        def f(blockIndex):
            blockWithHalo = blocking.getBlockWithHalo(blockIndex, margin)
            #print "fo"
            outerBlock = blockWithHalo.outerBlock
            outerSlicing = nifty.tools.getSlicing(outerBlock.begin, outerBlock.end)
            b,e = outerBlock.begin, outerBlock.end
            #print bi
            outerPmap = 1.0 - pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2],0]
            heightMap = pmapToHeightMap(outerPmap)

            with lock:
                done[0] += 1
                bar.update(done[0])


        nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=nWorkers)









with vigra.Timer("st"):
    params = {
        "axisResolution" :  [2.0, 2.0, 2.0],
        "featureBlockShape" : [200,200,200],
        "roiBegin": [0,0,0],
        "roiEnd":   [1000,1000,1000],
        #"nWorkers":1,
    }
    print pmapDset.shape
    membraneOverseg3D(pmapDset, **params)