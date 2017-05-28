import vigra
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.agglo
import numpy
import h5py
import sys

nrag = nifty.graph.rag
nagglo = nifty.graph.agglo

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

    features = []

    ucmFeat = ucmFeatures(raw=raw, pmap=pmap, 
                        overseg=overseg,rag=rag, 
                        settings=settings)
    features.append(ucmFeat)

    accFeat = accumulatedFeatures(raw=raw, pmap=pmap, 
                        overseg=overseg,rag=rag, 
                        settings=settings)
    features.append(accFeat)

    features = numpy.concatenate(features,axis=1)

    # save the features
    f5 = h5py.File(featuresFile, 'w') 
    f5['data'] = features
    f5.close()


@reraise_with_stack
def ucmFeatures(raw, pmap, overseg, rag, settings):
    
    features = []

    def ucmTransform(edgeIndicator, sizeRegularizer):
        vFeat = numpy.require(edgeIndicator, dtype='float32',requirements='C')
        eFeatures, nFeatures = nifty.graph.rag.accumulateMeanAndLength(rag, vFeat, [100,100],1)
        eMeans = eFeatures[:,0]
        eSizes = eFeatures[:,1]
        nSizes = nFeatures[:,1]
        clusterPolicy = nagglo.edgeWeightedClusterPolicyWithUcm(
            graph=rag, edgeIndicators=eMeans,
            edgeSizes=eSizes, nodeSizes=nSizes,
            numberOfNodesStop=1,
            sizeRegularizer=float(sizeRegularizer)
        )
        
        #print numpy.abs(eMeans-eMeansCp).sum() 
        #assert not numpy.array_equal(eMeans, eMeansCp)

        agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy)
        mergeHeightR = agglomerativeClustering.runAndGetDendrogramHeight()
        mergeHeight  = agglomerativeClustering.ucmTransform(clusterPolicy.edgeIndicators)
        mergeSize  = agglomerativeClustering.ucmTransform(clusterPolicy.edgeSizes)

       
        return mergeHeight,mergeHeightR,mergeSize

    for sigma in (1.0, 2.0, 4.0):
        for reg in (0.001, 0.1, 0.2, 0.4, 0.8):
            edgeIndicator = vigra.filters.hessianOfGaussianEigenvalues(raw, sigma)[:,:,0]

            A,B,C = ucmTransform(edgeIndicator,reg)

            features.append(A[:,None])
            features.append(B[:,None])
            features.append(C[:,None])

    for sigma in (0.1, 1.0, 2.0):
        for reg in (0.001, 0.1, 0.2, 0.4, 0.8):

            edgeIndicator = vigra.filters.gaussianSmoothing(pmap, sigma)
            A,B,C = ucmTransform(edgeIndicator,reg)

            features.append(A[:,None])
            features.append(B[:,None])
            features.append(C[:,None])

    return numpy.concatenate(features,axis=1)


@reraise_with_stack
def accumulatedFeatures(raw, pmap, overseg, rag, settings):
        
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

    for sigma in (1.0, 2.0, 4.0, 6.0 ,8.0):
        pf = [
            vigra.filters.hessianOfGaussianEigenvalues(raw, 1.0*sigma),
            vigra.filters.structureTensorEigenvalues(raw, 1.0*sigma, 2.0*sigma),
            vigra.filters.gaussianGradientMagnitude(raw, 1.0*sigma)[:,:,None],
            vigra.filters.gaussianSmoothing(raw, 1.0*sigma)[:,:,None]
        ]
        pixelFeats.extend(pf)

        if pmap is not None:
            pixelFeats.append(vigra.filters.gaussianSmoothing(pmap, 1.0*sigma)[:,:,None])
            pixelFeats.append(vigra.filters.hessianOfGaussianEigenvalues(pmap, 1.0*sigma)),
            pixelFeats.append(vigra.filters.structureTensorEigenvalues(pmap, 1.0*sigma, 2.0*sigma))

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

    return allEdgeFeat

