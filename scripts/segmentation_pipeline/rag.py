import vigra
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.agglo
import numpy
import h5py
import sys

from reraise import *

@reraise_with_stack
def makeRag(overseg, ragFile, settings):

    assert overseg.min() == 0 
    rag = nifty.graph.rag.gridRag(overseg)

    # serialize the rag
    serialization = rag.serialize()

    # save serialization
    f5 = h5py.File(ragFile, 'w') 
    f5['rag'] = serialization

    # save superpixels 
    f5['superpixels'] = overseg

    f5.close()




def loadRag(ragFile):
    # read serialization
    f5 = h5py.File(ragFile, 'r') 
    serialization = f5['rag'][:]

    # load superpixels 
    overseg = f5['superpixels'][:]

    f5.close()

    rag = nifty.graph.rag.gridRag(overseg, serialization=serialization)
    return rag, overseg




@reraise_with_stack
def localRagFeatures(raw, pmap, overseg, rag, featuresFile, settings):
        
    #print "bincoutn", numpy.bincount(overseg.reshape([-1])).size,"nNodes",rag.numberOfNodes

    uv = rag.uvIds()
    u  = uv[:,0]
    v  = uv[:,1]


    # geometric edge features
    geometricFeaturs = nifty.graph.rag.accumulateGeometricEdgeFeatures(rag,
                                                    blockShape=[75, 75],
                                                    numberOfThreads=1)

    allEdgeFeat = [geometricFeaturs]

    pixelFeats = [
        raw[:,:,None],
    ]
    if pmap is not None:
        pixelFeats.append(pmap[:,:,None])

    for sigma in (1.0, 2.0, 4.0):
        pf = [
            vigra.filters.hessianOfGaussianEigenvalues(raw, 1.0*sigma),
            vigra.filters.structureTensorEigenvalues(raw, 1.0*sigma, 2.0*sigma),
            vigra.filters.gaussianGradientMagnitude(raw, 1.0*sigma)[:,:,None],
            vigra.filters.gaussianSmoothing(raw, 1.0*sigma)[:,:,None]
        ]
        pixelFeats.extend(pf)

        if pmap is not None:
            pixelFeats.append(vigra.filters.gaussianSmoothing(pmap, 1.0*sigma)[:,:,None])

    pixelFeats = numpy.concatenate(pixelFeats, axis=2)

    

    for i  in range(pixelFeats.shape[2]):

        pixelFeat = pixelFeats[:,:,i]

        edgeFeat, nodeFeat = nifty.graph.rag.accumulateStandartFeatures(
            rag=rag, data=pixelFeat.astype('float32'),
            minVal=pixelFeat.min(),
            maxVal=pixelFeat.max(),
            blockShape=[75, 75],
            numberOfThreads=10
        )

        uFeat = nodeFeat[u,:]
        vFeat = nodeFeat[v,:]

        du = numpy.abs(edgeFeat,uFeat)
        dv = numpy.abs(edgeFeat,vFeat)


        fList =[
            uFeat + vFeat,
            uFeat * vFeat,
            numpy.abs(uFeat-vFeat),
            numpy.minimum(uFeat,vFeat),
            numpy.maximum(uFeat,vFeat),
            du + dv,
            numpy.abs(du-dv),
            numpy.minimum(du,dv),
            numpy.maximum(du,dv),
            edgeFeat
        ]

        edgeFeat = numpy.concatenate(fList, axis=1) 
        allEdgeFeat.append(edgeFeat)

    allEdgeFeat = numpy.concatenate(allEdgeFeat, axis=1) 

    #print allEdgeFeat.shape



    # save the features
    f5 = h5py.File(featuresFile, 'w') 
    f5['data'] = allEdgeFeat
    f5.close()