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
from tools import *




@reraise_with_stack
def multicutFromLocalProbs(raw, rag, localProbs, liftedEdges):


    u = liftedEdges[:,0]
    v = liftedEdges[:,1]

    # accumulate length (todo, implement function to just accumulate length)
    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, raw, [100,100],1)
    eSizes = eFeatures[:,1]
    eps = 0.0001
    clipped = numpy.clip(localProbs, eps, 1.0-eps)

    features = []

    for beta in (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):

        for powers in (0.0, 0.005, 0.1, 0.15, 0.2):

            # the weight of the weight 
            wWeight = eSizes**powers

            print "\n\nBETA ",beta
            w = numpy.log((1.0-clipped)/(clipped)) + numpy.log((1.0-beta)/(beta)) * wWeight
            obj = nifty.graph.multicut.multicutObjective(rag, w)


            factory = obj.multicutIlpCplexFactory()
            solver = factory.create(obj)
            visitor = obj.multicutVerboseVisitor()
            #ret = solver.optimize(visitor=visitor)
            ret = solver.optimize()

            res = ret[u] != ret[v]

            features.append(res[:,None])

    features = numpy.concatenate(features, axis=1)
    mean = numpy.mean(features, axis=1)[:,None]
    features = numpy.concatenate([features, mean], axis=1)
    
    return features


@reraise_with_stack
def ucmFromLocalProbs(raw, rag, localProbs, liftedEdges, liftedObj):


    u = liftedEdges[:,0]
    v = liftedEdges[:,1]

    # accumulate length (todo, implement function to just accumulate length)
    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, raw, [100,100],1)
    eSizes = eFeatures[:,1]
    nSizes = eFeatures[:,1]


    feats = nifty.graph.lifted_multicut.liftedUcmFeatures(
       objective=liftedObj,
       edgeIndicators=localProbs,
       edgeSizes=eSizes,
       nodeSizes=nSizes,
       sizeRegularizers=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                         0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 
                         0.9, 0.95]
    )

    return feats



@reraise_with_stack
def ucmFromHessian(raw, rag, liftedEdges, liftedObj):


    u = liftedEdges[:,0]
    v = liftedEdges[:,1]

    feats = []



    for sigma in [1.0, 2.0, 3.0, 4.0, 5.0]:
        pf = vigra.filters.hessianOfGaussianEigenvalues(raw, sigma)[:,:,0]
        eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, pf, [100,100],1)
        edgeIndicator =  eFeatures[:,0]
        eSizes = eFeatures[:,1]
        nSizes = eFeatures[:,1]

        featsB = nifty.graph.lifted_multicut.liftedUcmFeatures(
           objective=liftedObj,
           edgeIndicators=edgeIndicator,
           edgeSizes=eSizes,
           nodeSizes=nSizes,
           sizeRegularizers=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                             0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 
                             0.9, 0.95]
        )

        feats.append(featsB)

    # different axis ordering as usual
    return numpy.concatenate(feats,axis=0)








@reraise_with_stack
def liftedFeaturesFromLocalProbs(raw, rag, localProbs, liftedEdges, liftedObj, featureFile):
    

    mcFeatureFile = featureFile + "mc.h5"
    if not hasH5File(mcFeatureFile):
        mcFeatures = multicutFromLocalProbs(raw=raw, rag=rag, localProbs=localProbs, 
                                          liftedEdges=liftedEdges)

        f5 = h5py.File(mcFeatureFile, 'w') 
        f5['data'] = mcFeatures
        f5.close()
    else:
        mcFeatures = h5Read(mcFeatureFile)

    
    ucmFeatureFile = featureFile + "ucm.h5"
    if not hasH5File(ucmFeatureFile):
        ucmFeatures = ucmFromLocalProbs(raw=raw, rag=rag, localProbs=localProbs, 
                                          liftedEdges=liftedEdges,
                                          liftedObj=liftedObj)

        f5 = h5py.File(ucmFeatureFile, 'w') 
        f5['data'] = ucmFeatures
        f5.close()
    else:
        ucmFeatures = h5Read(ucmFeatureFile)



    # combine
    features = numpy.concatenate([mcFeatures, ucmFeatures.swapaxes(0,1)],axis=1)

    f5 = h5py.File(featureFile, 'w') 
    f5['data'] = features
    f5.close()





@reraise_with_stack
def accumlatedLiftedFeatures(raw, pmap, rag, liftedEdges, liftedObj):


    uv = liftedEdges
    u  = uv[:,0]
    v  = uv[:,1]


    # geometric edge features
    geometricFeaturs = nifty.graph.rag.accumulateGeometricNodeFeatures(rag,
                                                    blockShape=[75, 75],
                                                    numberOfThreads=1)

    nodeSize = geometricFeaturs[:,0]
    nodeCenter = geometricFeaturs[:,1:2]
    nodeAxisA = geometricFeaturs[:,2:4]
    nodeAxisA = geometricFeaturs[:,4:6]

    diff = (nodeCenter[u,:] - nodeCenter[v,:])**2
    diff = numpy.sum(diff,axis=1)




    allEdgeFeat = [
        # sizes
        (nodeSize[u] + nodeSize[v])[:,None],
        (numpy.abs(nodeSize[u] - nodeSize[v]))[:,None],
        (nodeSize[u] * nodeSize[v])[:,None],
        (numpy.minimum(nodeSize[u] , nodeSize[v]))[:,None],
        (numpy.maximum(nodeSize[u] , nodeSize[v]))[:,None],
        diff[:,None]
    ]

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

        nodeFeat = nifty.graph.rag.accumulateNodeStandartFeatures(
            rag=rag, data=pixelFeat.astype('float32'),
            minVal=pixelFeat.min(),
            maxVal=pixelFeat.max(),
            blockShape=[75, 75],
            numberOfThreads=10
        )

        uFeat = nodeFeat[u,:]
        vFeat = nodeFeat[v,:]


        fList =[
            uFeat + vFeat,
            uFeat * vFeat,
            numpy.abs(uFeat-vFeat),
            numpy.minimum(uFeat,vFeat),
            numpy.maximum(uFeat,vFeat),
        ]

        edgeFeat = numpy.concatenate(fList, axis=1) 
        allEdgeFeat.append(edgeFeat)

    allEdgeFeat = numpy.concatenate(allEdgeFeat, axis=1) 

    return allEdgeFeat



@reraise_with_stack
def liftedFeatures(raw, pmap, rag, liftedEdges, liftedObj, distances, featureFile):
    


    
    ucmFeatureFile = featureFile + "ucm.h5"
    if not hasH5File(ucmFeatureFile):
        ucmFeatures = ucmFromHessian(raw=raw, rag=rag, 
                                    liftedEdges=liftedEdges,
                                    liftedObj=liftedObj)

        f5 = h5py.File(ucmFeatureFile, 'w') 
        f5['data'] = ucmFeatures
        f5.close()
    else:
        ucmFeatures = h5Read(ucmFeatureFile)


    accFeatureFile = featureFile + "acc.h5"
    if not hasH5File(accFeatureFile):
        accFeatrues = accumlatedLiftedFeatures(raw=raw, pmap=pmap, 
                                    rag=rag, 
                                    liftedEdges=liftedEdges,
                                    liftedObj=liftedObj)

        f5 = h5py.File(accFeatureFile, 'w') 
        f5['data'] = accFeatrues
        f5.close()
    else:
        accFeatrues = h5Read(accFeatureFile)


    # combine
    features = numpy.concatenate([accFeatrues,distances[:,None], ucmFeatures.swapaxes(0,1)],axis=1)

    f5 = h5py.File(featureFile, 'w') 
    f5['data'] = features
    f5.close()