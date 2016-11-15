from __future__ import print_function
from __future__ import division

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

pmapPath = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/hhess_stuart/A_no_border_pmap.h5"
rawFile = None


heightMapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/hhess_stuart/A_heightMap.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/hhess_stuart/A_newoverseg2.h5"
agglosegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/hhess_stuart/A_new_aggloseg.h5"


pmapH5 = h5py.File(pmapPath,'r')
pmapDset = pmapH5['data']





def makeBall(r):
    size = 2*r + 1

    mask = numpy.zeros([size]*3)

    for x0 in range(-1*r, r + 1):
        for x1 in range(-1*r, r + 1):
            for x2 in range(-1*r, r + 1):
                
                if math.sqrt(x0**2 + x1**2 + x2**2) <= r:
                    mask[x0+r, x1+r, x2+r] = 1

    return mask, (r,r,r)


def membraneOverseg3D(pmapDset, heightMapDset, **kwargs):



    axisResolution = kwargs.get("axisResolution",['4nm']*3)
    featureBlockShape = kwargs.get("featureBlockShape",['100']*3)
    shape = pmapDset.shape[0:3]
    

    roiBegin = kwargs.get("roiBegin", [0]*3)
    roiEnd = kwargs.get("roiEnd", shape)
    nWorkers = kwargs.get("nWorkers",cpu_count())
    invertPmap = kwargs.get("invertPmap",False)

    blocking = nifty.tools.blocking(roiBegin=roiBegin, roiEnd=roiEnd, blockShape=featureBlockShape)
    margin = [45 ,45,45]


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

        combined = medianImg + blurredSuperLarge*0.3 + 0.15*blurredLarge + 0.1*blurredSmall

        footprint, origin = makeBall(r=3)
        combined = scipy.ndimage.percentile_filter(input=combined, 
                                                    #size=(20,20,20),
                                                    footprint=footprint, 
                                                    #origin=origin, 
                                                    mode='reflect',
                                                    percentile=50.0)


        if False:
            nifty.viewer.view3D(pmap, show=False, title='pm',cmap='gray')
            nifty.viewer.view3D(medianImg, show=False, title='medianImg',cmap='gray')
            nifty.viewer.view3D(combined, show=False, title='combined',cmap='gray')
            pylab.show()

        return combined


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
        
            if invertPmap:
                outerPmap = 1.0 - pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
            else:
                outerPmap = pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

            heightMap = pmapToHeightMap(outerPmap)

            # 
            innerBlockLocal = blockWithHalo.innerBlockLocal
            b,e = innerBlockLocal.begin, innerBlockLocal.end
            innerHeightMap = heightMap[b[0]:e[0], b[1]:e[1], b[2]:e[2]]


            b,e =  blockWithHalo.innerBlock.begin,  blockWithHalo.innerBlock.end

            if isinstance(heightMapDset,numpy.ndarray):
                heightMapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = innerHeightMap
                with lock:
                    done[0] += 1
                    bar.update(done[0])

            else:
                with lock:
                    heightMapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = innerHeightMap
                    done[0] += 1
                    bar.update(done[0])


        nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=nWorkers)






def makeSmallerSegNifty(oseg,  volume_feat, reduceBy, wardness):
    import nifty
    import nifty.graph
    import nifty.graph.rag
    import nifty.graph.agglo

    nrag = nifty.graph.rag
    nagglo = nifty.graph.agglo

    print("overseg in c order starting at zero")
    oseg = numpy.require(oseg, dtype='uint32',requirements='C')
    oseg -= 1

    print("make rag")
    rag = nifty.graph.rag.gridRag(oseg)

    print("volfeatshape")
    vFeat = numpy.require(volume_feat, dtype='float32',requirements='C')

    print("accumulate means and counts")
    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, vFeat, [100,100,100],-1)

    eMeans = eFeatures[:,0]
    eSizes = eFeatures[:,1]
    nSizes = nFeatures[:,1]

    print("get clusterPolicy")

    numberOfNodesStop = int(float(rag.numberOfNodes)/float(reduceBy) + 0.5)
    numberOfNodesStop = max(1,numberOfNodesStop)
    numberOfNodesStop = min(rag.numberOfNodes, numberOfNodesStop)

    clusterPolicy = nagglo.edgeWeightedClusterPolicy(
        graph=rag, edgeIndicators=eMeans,
        edgeSizes=eSizes, nodeSizes=nSizes,
        numberOfNodesStop=numberOfNodesStop,
        sizeRegularizer=float(wardness))

    print("do clustering")
    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run()
    seg = agglomerativeClustering.result()#out=[1,2,3,4])

    print("make seg dense")
    dseg = nifty.tools.makeDense(seg)

    print(dseg.dtype, type(dseg))

    print("project to pixels")
    pixelData = nrag.projectScalarNodeDataToPixels(rag, dseg.astype('uint32'))
    print("done")
    #pixelDataF = numpy.require(pixelData, dtype='uint32',requirements='F')
    return pixelData+1
    


if False:
    pmapH5 = h5py.File(pmapPath,'r')
    pmapDset = pmapH5['data']

    heightMapH5 = h5py.File(heightMapFile,'w')
    heightMapDset = heightMapH5.create_dataset('data',shape=pmapDset.shape, chunks=(100,100,100),dtype='float32')

    with vigra.Timer("st"):
        params = {
            "axisResolution" :  [2.0, 2.0, 2.0],
            "featureBlockShape" : [200,200,200],
            "invertPmap": False,
            #"roiBegin": [0,0,0],
            #"roiEnd":   [1000,1000,1000],
            #"nWorkers":1,
        }
        print(pmapDset.shape)
        membraneOverseg3D(pmapDset,heightMapDset, **params)



    heightMapH5.close()
    pmapH5.close()

if False:


    heightMapH5 = h5py.File(heightMapFile,'r')
    heightMapDset = heightMapH5['data']

    oversegH5 = h5py.File(oversegFile,'w')
    oversegDset = oversegH5.create_dataset('data',shape=pmapDset.shape, chunks=(100,100,100),dtype='uint32')

    print("read hmap")
    heightMap = heightMapDset[:,:,:]

    print("do overseg")
    overseg,= vigra.analysis.unionFindWatershed3D(heightMap.T, blockShape=(100,100,100))


    print("transpose")
    overseg = overseg.T

    print("write")
    oversegDset[:,:,:] = overseg
    oversegH5.close()
    pmapH5.close()


if True:


    heightMapH5 = h5py.File(heightMapFile,'r')
    heightMapDset = heightMapH5['data']

    oversegH5 = h5py.File(oversegFile,'r')
    oversegDset = oversegH5['data']

    agglosegH5 = h5py.File(agglosegFile,'w')
    agglosegDset = agglosegH5.create_dataset('data',shape=pmapDset.shape, chunks=(100,100,100),dtype='uint32')


    print("read hmap")
    heightMap = heightMapDset[:,:,:]

    print("read oseg")
    overseg = oversegDset[:,:,:]

    print("make smaller")
    smallerSeg = makeSmallerSegNifty(overseg,heightMap, 20, 0.3)

    print("write")
    agglosegDset[:,:,:] = smallerSeg



    oversegH5.close()
    pmapH5.close()
    agglosegH5.close()