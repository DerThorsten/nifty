import vigra
import nifty
import numpy
import random
import os
import h5py

from sklearn.cross_validation import KFold
import xgboost as xgb


from tools import *
from superpixels import *
from rag import *
from gt import *
from lifted_features import *


assert nifty.Configuration.WITH_CPLEX

def runPipeline(trainInput, testInput, settings):
    


    random.seed(42)

    # prepare data
    trainRaw = numpy.array(vigra.impex.readVolume(trainInput['rawData'])).squeeze()
    trainGt  = numpy.array(vigra.impex.readVolume(trainInput['gt'])).squeeze()
    testRaw  = numpy.array(vigra.impex.readVolume(testInput['rawData'])).squeeze()

    # prob. map
    trainPmap = numpy.array(h5Read(*trainInput['pmap'])).squeeze()
    testPmap = numpy.array(h5Read(*testInput['pmap'])).squeeze()

    # superpixels
    trainSp = numpy.array(h5Read(*trainInput['superpixels'])).squeeze()
    testSp = numpy.array(h5Read(*testInput['superpixels'])).squeeze()




    trainRaw = numpy.rollaxis(trainRaw, 2, 0 )#[0:6,:,:]
    testRaw = numpy.rollaxis(testRaw, 2, 0 )  #[0:6,:,:]
    
    trainPmap = numpy.rollaxis(trainPmap, 2, 0 )#[0:6,:,:]
    testPmap = numpy.rollaxis(testPmap, 2, 0 )  #[0:6,:,:]

    trainSp = numpy.rollaxis(trainSp, 2, 0 )#[0:6,:,:]
    testSp = numpy.rollaxis(testSp, 2, 0 )  #[0:6,:,:]   




    # vigra.segShow(testRaw[0,:,:],testSp[0,:,:])
    # vigra.show()

    # let supervoxels start at zero
    for sps in [trainSp,testSp]:
        for s in range(sps.shape[0]):
            sps[s,:,:] -= sps[s,:,:].min()                                      
                                                                       
    trainGt = numpy.rollaxis(trainGt, 2, 0 )#[0:6,:,:]
    


    # prepare output folders
    rootOutDir = settings['rootOutDir']




    trainTestDict = {
       'train' : {
           'input' : trainInput,
           'raw'   : trainRaw,
           'pmap'  : trainPmap,
           'sp'    : trainSp,
           'gt'    : trainGt
        }
        ,
        'test' : {
        'input' : testInput,
        'raw'   : testRaw,
        'pmap'  : testPmap,
        'sp'    : testSp
        }
    }

    # create outfolders
    for key in   trainTestDict.keys():
        outDir = os.path.join(rootOutDir, key)
        print outDir
        ensureDir(outDir)
        trainTestDict[key]['outDir'] = outDir


    # prepare the ground truth
    prepareGroundTruth(trainTestDict['train'], settings)

    # init  superpixels 
    # either load them or compute them
    # in both cases they are stored at a common place
    #for key in trainTestDict:
    #    computeSuperpixels(trainTestDict[key], settings)


    # init region adjacency graph
    for key in trainTestDict:
        computeRag(trainTestDict[key], settings)

    # compute rag features
    for key in trainTestDict:
        computeRagFeatures(trainTestDict[key], settings)


    # generate subsets 
    # ensure that these random number are the same for each run
    numpy.random.seed(42)
    splits = generateSplits(trainRaw.shape[0], nSplits=3,frac=0.33)
    splits.extend(generateSplits(trainRaw.shape[0], nSplits=3,frac=0.66))
    splits.extend(generateSplits(trainRaw.shape[0], nSplits=3,frac=0.50))



    # train local classifier(s)
    for splitIndex, (trainLocal, notTrainLocal) in enumerate(splits):
        trainLocalRf(trainTestDict['train'], settings, subset=trainLocal, clfName=str(splitIndex))
    trainLocalRf(trainTestDict['train'], settings, subset=trainLocal, clfName='all')


    # predict local classifier(s) probs
    for splitIndex, (trainLocal, notTrainLocal) in enumerate(splits):
        predictLocalRf(trainTestDict['train'], settings, subset=notTrainLocal, clfName=str(splitIndex))

    # predict on test set
    for splitIndex, (_,_) in enumerate(splits):
        predictLocalRf(trainTestDict['test'], settings, subset=None, clfName=str(splitIndex))
    predictLocalRf(trainTestDict['test'], settings, subset=None, clfName='all')

    #################################################
    #  LIFTED GRAPH
    #################################################
    
    # get the additional lifted edges
    for key in trainTestDict:
        getLiftedEdges(trainTestDict[key], settings)


    for key in trainTestDict:
        computeLiftedFeatures(trainTestDict[key], settings)


    # get features for lifted graph wich depend on the local rag probs
    for splitIndex, (trainLocal,notTrainLocal) in enumerate(splits):
        computeLiftedFeaturesFromLocalProbs(trainTestDict['train'], settings, subset=notTrainLocal, clfName=str(splitIndex))

    for splitIndex, (trainLocal,notTrainLocal) in enumerate(splits):
        computeLiftedFeaturesFromLocalProbs(trainTestDict['test'], settings, subset=None, clfName=str(splitIndex))


    # train lifted mc classifier(s)
    for splitIndex, (trainLocal,notTrainLocal) in enumerate(splits):
        trainLiftedClf(trainTestDict['train'], settings, subset=notTrainLocal, clfName=str(splitIndex))

    clfNames = []
    clfWeights = []
    for splitIndex, (trainLocal,notTrainLocal) in enumerate(splits):
        clfNames.append(str(splitIndex))
        clfWeights.append(len(notTrainLocal))

    # do the prediction
    predictLifted(trainTestDict['test'], settings, clfNames, clfWeights)

    # run the lifted mc on the test set
    runLiftedMc(trainTestDict['test'], settings)


def prepareGroundTruth(dataDict, settings):

    rawData = dataDict['raw']
    gtData = dataDict['gt']
    outDir = dataDict['outDir']

    gtFile = os.path.join(outDir,'groundTruth.h5')

    if not hasH5File(gtFile):




        gt = numpy.empty_like(rawData,dtype='uint32')

        futures = []
        with  threadExecutor() as executor:
            for sliceIndex  in range(rawData.shape[0]):
                binaryGt = numpy.array(gtData[sliceIndex, :, :])
                rd = numpy.array(rawData[sliceIndex, :, :])
                future = executor.submit(makeGt, rd, binaryGt, settings)
                futures.append(future)

        for sliceIndex  in range(rawData.shape[0]):
            gt[sliceIndex,:,:] = futures[sliceIndex].result()


        # if settings['debug']:
        #    vigra.segShow(rawData[5,:,:], gt[5,:,:])
        #    vigra.show()

        # store supervoxels in common place
        hfile = h5py.File(os.path.join(outDir,'groundTruth.h5'),'w')
        hfile['data'] = gt
        hfile.close()

def computeSuperpixels(dataDict, settings):

    rawData = dataDict['raw']
    spH5Path = dataDict['input']['superpixels']
    outDir = dataDict['outDir']

    spFile = os.path.join(outDir,'supervoxels.h5')

    if not hasH5File(spFile):

        # user has provided supervoxels
        if spH5Path is not None: 
            if not isH5Path(spH5Path):
                raise RuntimeError("Superpixel path is no h5Path:\n Must be a tuple like: ('file.h5','dset')")
            else:
                f,d = spH5Path
                h5File = h5py.File(f)
                superpixels = h5File[d][:].squeeze()
                h5File.close()

                if(superpixels.shape != rawData.shape):
                    raise RuntimeError("Provided Superpixels have wrong shape: rawDataShape: \
                                        rawDataShape: %s , supervoxelsShape: %s" \
                                        % (str(rawData.shape), str(superpixels.shape)))



        # user has not provided supervoxels
        else:
            printWarning("WARNING:\nyou should provide your own superpixels based on your pmaps")
            

            superpixels = numpy.empty_like(rawData,dtype='uint32')

            futures = []
            with  threadExecutor() as executor:
                for sliceIndex  in range(rawData.shape[0]):
                    rd = numpy.array(rawData[sliceIndex, :, :])
                    future = executor.submit(makeSupervoxels, rd, settings)
                    futures.append(future)

            for sliceIndex  in range(rawData.shape[0]):
                superpixels[sliceIndex,:,:] = futures[sliceIndex].result()


            #if settings['debug']:
            #    vigra.segShow(rawData[5,:,:], superpixels[5,:,:])
            #    vigra.show()

        # store supervoxels in common place
        hfile = h5py.File(os.path.join(outDir,'supervoxels.h5'),'w')
        hfile['data'] = superpixels
        hfile.close()



def getGt(dataDict, settings):
    outDir = dataDict['outDir']
    if 'neuronGt' in dataDict:
        return dataDict['neuronGt']
    else:
        gtFile = os.path.join(outDir,'groundTruth.h5')
        gt =  h5Read(gtFile)
        dataDict['neuronGt'] = gt
        return  gt

def computeRag(dataDict, settings):

    rawData = dataDict['raw']
    superpixels = dataDict['sp']
    outDir = dataDict['outDir']
    ragDir =  os.path.join(outDir,'rag')
    ensureDir(ragDir)

    futures = []
    with  threadExecutor() as executor:
        for sliceIndex  in range(rawData.shape[0]):
            
            
            ragFile = os.path.join(ragDir,'rag_%d.h5'%sliceIndex)
            if not hasH5File(ragFile):

                overseg = superpixels[sliceIndex, :,:]
                #makeRag(overseg, ragFile, settings)
                future = executor.submit(makeRag, overseg, ragFile, settings)
                futures.append(future)
            else:
                pass

    for f in futures:
        f.result()
    
def getRagsAndSuperpixels(dataDict, settings):

    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragDir =  os.path.join(outDir,'rag')

    if 'ragsAndSuperpixels' in dataDict:
        return dataDict['ragsAndSuperpixels']
    else:
        rags = []

        for sliceIndex  in range(rawData.shape[0]):
            ragFile = os.path.join(ragDir,'rag_%d.h5'%sliceIndex)

            rag,sp = loadRag(ragFile)

            rags.append((rag,sp))

        dataDict['ragsAndSuperpixels'] = rags
        return rags

def computeRagFeatures(dataDict, settings):

    rawData = dataDict['raw']
    pmap    = dataDict['pmap']

    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')
    ensureDir(ragFeatDir)

    futures = []
    with  threadExecutor() as executor:
        for sliceIndex  in range(rawData.shape[0]):
            
            
            ragFeatFile = os.path.join(ragFeatDir,'rag_feat_%d.h5'%sliceIndex)
            if not hasH5File(ragFeatFile):

                rag,sp = ragsAndSuperpixels[sliceIndex]

                localRagFeatures(
                    raw=rawData[sliceIndex,:,:], 
                    pmap=pmap[sliceIndex,:,:],
                    overseg=sp, rag=rag, featuresFile=ragFeatFile,
                    settings=settings)

                #future = executor.submit(localRagFeatures,
                #    raw=rawData[sliceIndex,:,:], pmap=None,
                #    overseg=sp, rag=rag, featuresFile=ragFeatFile,
                #    settings=settings)
                #futures.append(future)
            else:
                pass

    for f in futures:
        f.result()

def trainLocalRf(dataDict, settings,clfName, subset=None):

    

    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')
    gt = getGt(dataDict, settings)
    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)

    fname = os.path.join(settings['rootOutDir'], "local_clf_%s"%clfName)

    if not hasFile(fname):

        X = []
        Y = []

        for sliceIndex  in range(rawData.shape[0]):


            if subset is None or sliceIndex in subset:

                # feature file
                ragFeatFile = os.path.join(ragFeatDir,'rag_feat_%d.h5'%sliceIndex)
                features = h5Read(ragFeatFile)

                # rag and sp
                rag, sp = ragsAndSuperpixels[sliceIndex]
                # pixel wise gt
                pixelWiseGt = gt[sliceIndex,:,:]

                features, labels = getTrainingData(rag=rag, sp=sp, pixelGt=pixelWiseGt, 
                                features=features, settings=settings)

                X.append(features)
                Y.append(labels)

        X = numpy.concatenate(X,axis=0)
        Y = numpy.concatenate(Y,axis=0)


        clfSettings = settings['clfLocal']

        clf = Classifier(nRounds=clfSettings['nRounds'], maxDepth=clfSettings['maxDepth'])
        clf.train(X=X, Y=Y,getApproxError=clfSettings['getApproxError'])
        clf.save(fname=fname)

        

def predictLocalRf(dataDict, settings, clfName, subset):

    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')

    localRfPredictDir =  os.path.join(outDir,'localClfProbs')
    ensureDir(localRfPredictDir)

    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    fname = os.path.join(settings['rootOutDir'], "local_clf_%s"%clfName)


    bst = xgb.Booster({'nthread':10}) #init model
    bst.load_model(fname)             # load data
    
    clf = Classifier()
    clf.load(fname)



    for sliceIndex  in range(rawData.shape[0]):


            if subset is None or sliceIndex in subset:

                
                predictionFile = os.path.join(localRfPredictDir,'rag_pred_clf%s_%d.h5'%(clfName, sliceIndex))

                if not hasH5File(predictionFile):
                    # feature file
                    ragFeatFile = os.path.join(ragFeatDir,'rag_feat_%d.h5'%sliceIndex)
                    features = h5Read(ragFeatFile)

                    ypred = clf.predict(features)

                    # save the predictions
                    f5 = h5py.File(predictionFile, 'w') 
                    f5['data'] = ypred
                    f5.close()

def getLiftedEdges(dataDict, settings):
    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')

    liftedEdgesDir =  os.path.join(outDir,'liftedEdges')
    ensureDir(liftedEdgesDir)

    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)

    for sliceIndex  in range(rawData.shape[0]):

        liftedEdgesFile = os.path.join(liftedEdgesDir,'lifted_edges_%d.h5'%(sliceIndex))

        if not hasH5File(liftedEdgesFile):

            rag, sp = ragsAndSuperpixels[sliceIndex]

            obj = nifty.graph.lifted_multicut.liftedMulticutObjective(rag)
            liftedGraph = obj.liftedGraph

            distance = obj.insertLiftedEdgesBfs(5, returnDistance=True)
            liftedUvIds = obj.liftedUvIds()

            assert distance.shape[0] == liftedUvIds.shape[0]

            print rag.numberOfEdges, liftedGraph.numberOfEdges  


            f5 = h5py.File(liftedEdgesFile, 'w') 
            f5['liftedUvIds'] = liftedUvIds
            f5['distance'] = distance
            f5.close()


def computeLiftedFeatures(dataDict, settings):
    print "computeLiftedFeatures"
    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')

    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    liftedEdgesDir =  os.path.join(outDir,'liftedEdges')
    liftedFeaturesDir =  os.path.join(outDir,'liftedFeatures')
    ensureDir(liftedFeaturesDir)


    for sliceIndex  in range(rawData.shape[0]):
      
            
        liftedEdgesFile = os.path.join(liftedEdgesDir,'lifted_edges_%d.h5'%(sliceIndex))
        liftedFeatureFile = os.path.join(liftedFeaturesDir,'lifted_features_%d.h5'%(sliceIndex))


        if not hasH5File(liftedFeatureFile):

            rag, sp = ragsAndSuperpixels[sliceIndex]
            liftedEdges = h5Read(liftedEdgesFile, 'liftedUvIds')

            obj = nifty.graph.lifted_multicut.liftedMulticutObjective(rag)
            liftedGraph = obj.liftedGraph

            distances = obj.insertLiftedEdgesBfs(5, returnDistance=True)
            liftedUvIds = obj.liftedUvIds()


            liftedFeatures(raw=rawData[sliceIndex,:,:],pmap=None,
                rag=rag, liftedEdges=liftedEdges,
                liftedObj=obj,distances=distances,
                featureFile=liftedFeatureFile)

def computeLiftedFeaturesFromLocalProbs(dataDict, settings, clfName, subset):
    print "computeLiftedFeaturesFromLocalProbs"
    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    ragFeatDir =  os.path.join(outDir,'ragFeatures')
    localRfPredictDir =  os.path.join(outDir,'localClfProbs')
    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    liftedEdgesDir =  os.path.join(outDir,'liftedEdges')
    liftedFeaturesDir =  os.path.join(outDir,'liftedFeaturesFromLocalProbs')
    ensureDir(liftedFeaturesDir)


    for sliceIndex  in range(rawData.shape[0]):
        if subset is None or sliceIndex in subset:
            
            predictionFile = os.path.join(localRfPredictDir,'rag_pred_clf%s_%d.h5'%(clfName, sliceIndex))
            liftedEdgesFile = os.path.join(liftedEdgesDir,'lifted_edges_%d.h5'%(sliceIndex))
            liftedFeatureFile = os.path.join(liftedFeaturesDir,'lifted_features_clf%s_%d.h5'%(clfName, sliceIndex))


            if not hasH5File(liftedFeatureFile):

                rag, sp = ragsAndSuperpixels[sliceIndex]
                localProbs = h5Read(predictionFile)[:,1]
                liftedEdges = h5Read(liftedEdgesFile, 'liftedUvIds')



                obj = nifty.graph.lifted_multicut.liftedMulticutObjective(rag)
                liftedGraph = obj.liftedGraph

                distance = obj.insertLiftedEdgesBfs(5, returnDistance=True)
                liftedUvIds = obj.liftedUvIds()


                liftedFeaturesFromLocalProbs(raw=rawData[sliceIndex,:,:], 
                    rag=rag, localProbs=localProbs,
                    liftedEdges=liftedEdges,
                    liftedObj=obj,
                    featureFile=liftedFeatureFile)

def trainLiftedClf(dataDict, settings, clfName, subset):

    print "trainLiftedClf"
    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    gt = getGt(dataDict, settings)
    ragFeatDir =  os.path.join(outDir,'ragFeatures')
    localRfPredictDir =  os.path.join(outDir,'localClfProbs')
    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    liftedEdgesDir =  os.path.join(outDir,'liftedEdges')
    liftedFeaturesFromLocalDir =  os.path.join(outDir,'liftedFeaturesFromLocalProbs')
    liftedFeaturesDir =  os.path.join(outDir,'liftedFeatures')



    fname = os.path.join(settings['rootOutDir'], "lifted_clf_%s"%clfName)

    if not hasFile(fname):

        X = []
        Y = []



        for sliceIndex  in range(rawData.shape[0]):
            if subset is None or sliceIndex in subset:
                


                liftedEdgesFile = os.path.join(liftedEdgesDir,'lifted_edges_%d.h5'%(sliceIndex))
                liftedFeatureFileFromProbs = os.path.join(liftedFeaturesFromLocalDir,'lifted_features_clf%s_%d.h5'%(clfName, sliceIndex))
                liftedFeatureFile = os.path.join(liftedFeaturesDir,'lifted_features_%d.h5'%(sliceIndex))

                rag, sp = ragsAndSuperpixels[sliceIndex]
                liftedEdges = h5Read(liftedEdgesFile, 'liftedUvIds')


                obj = nifty.graph.lifted_multicut.liftedMulticutObjective(rag)
                liftedGraph = obj.liftedGraph

                distance = obj.insertLiftedEdgesBfs(5, returnDistance=True)
                liftedUvIds = obj.liftedUvIds()




                # feature file
                featuresFromProbs = h5Read(liftedFeatureFileFromProbs)
                features = h5Read(liftedFeatureFile)
                features = numpy.concatenate([featuresFromProbs, features], axis=1)

                print "Features",features.shape
                # rag and sp
                rag, sp = ragsAndSuperpixels[sliceIndex]

                # pixel wise gt
                pixelWiseGt = gt[sliceIndex,:,:]

                features, labels = getLiftedTrainingData(rag=rag, sp=sp, pixelGt=pixelWiseGt, 
                                liftedUvIds=liftedUvIds,
                                features=features, settings=settings)

                X.append(features)
                Y.append(labels)
        

        X = numpy.concatenate(X,axis=0)
        Y = numpy.concatenate(Y,axis=0)

        print X.shape, Y.shape

        clfSettings = settings['clfLifted']
        clf = Classifier(nRounds=clfSettings['nRounds'], maxDepth=clfSettings['maxDepth'])
        trainError = clf.train(X=X, Y=Y,getApproxError=clfSettings['getApproxError'])    
        print "trainError",trainError
        clf.save(fname=fname)




def predictLifted(dataDict, settings, clfNames, clfWeights):

    print "predictLifted"
    rawData = dataDict['raw']
    outDir = dataDict['outDir']
   
    liftedFeaturesFromLocalDir =  os.path.join(outDir,'liftedFeaturesFromLocalProbs')
    liftedFeaturesDir =  os.path.join(outDir,'liftedFeatures')

    liftedProbsDir =  os.path.join(outDir,'liftedProbs')
    ensureDir(liftedProbsDir)

    clfs = []

    for clfName in clfNames:
        fname = os.path.join(settings['rootOutDir'], "lifted_clf_%s"%clfName)
        clf = Classifier()
        clf.load(fname=fname)
        clfs.append(clf)



    for sliceIndex  in range(rawData.shape[0]):
       

            # lifted probs file
            liftedProbsFile = os.path.join(liftedProbsDir,'lifted_probs_%d.h5'%(sliceIndex))

            if not hasH5File(liftedProbsFile):

                liftedFeatureFileFromProbs = os.path.join(liftedFeaturesFromLocalDir,'lifted_features_clf%s_%d.h5'%(clfName, sliceIndex))
                liftedFeatureFile = os.path.join(liftedFeaturesDir,'lifted_features_%d.h5'%(sliceIndex))

                

                # feature file
                featuresFromProbs = h5Read(liftedFeatureFileFromProbs)
                features = h5Read(liftedFeatureFile)
                features = numpy.concatenate([featuresFromProbs, features], axis=1)

                
                probs = numpy.zeros([features.shape[0]],dtype='float32')
                totalW = 0.0
                for clf,w in zip(clfs,clfWeights):
                    probs += clf.predict(features)[:,1] * w
                    totalW += w

                probs /= totalW

                whereBig = numpy.where(probs>0.7)[0]
        

                f5 = h5py.File(liftedProbsFile, 'w') 
                f5['data'] = probs
                f5.close()


def runLiftedMc(dataDict, settings):


    rawData = dataDict['raw']
    outDir = dataDict['outDir']
    localRfPredictDir  =  os.path.join(outDir,'localClfProbs')
    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)
    liftedFeaturesDir  =  os.path.join(outDir,'liftedFeatures')
    liftedProbsDir =  os.path.join(outDir,'liftedProbs')

    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)

    for sliceIndex  in range(rawData.shape[0]):


        # local probs
        predictionFile = os.path.join(localRfPredictDir,'rag_pred_clf%s_%d.h5'%('0', sliceIndex))
        localProbs = h5Read(predictionFile)[:,1]
    
        # lifted probs
        liftedProbsFile = os.path.join(liftedProbsDir,'lifted_probs_%d.h5'%(sliceIndex))
        liftedProbs = h5Read(liftedProbsFile)

        # set up the lifted objective
        rag, sp = ragsAndSuperpixels[sliceIndex]
        obj = nifty.graph.lifted_multicut.liftedMulticutObjective(rag)
        liftedGraph = obj.liftedGraph
        distance = obj.insertLiftedEdgesBfs(5, returnDistance=True).astype('float32')
        liftedUvIds = obj.liftedUvIds()

        # numeric factor to not get to small weights
        # might not be necessary
        C = 100.0

        # local weights
        eps = 0.0001
        clipped = numpy.clip(localProbs, eps, 1.0-eps)
        beta = settings['betaLocal']
        wLocal = numpy.log((1.0-clipped)/(clipped)) + numpy.log((1.0-beta)/(beta))
        # normalize by number of local edges length
        wLocal *= C/len(wLocal)
        wLocal *= settings['gamma']

        # non local weights
        eps = 0.0001
        clipped = numpy.clip(liftedProbs, eps, 1.0-eps)
        beta = settings['betaLifted']
        wLifted = numpy.log((1.0-clipped)/(clipped)) + numpy.log((1.0-beta)/(beta))
       

        # normalize by the distance (give close one more weight)
        distanceWeight = 1.0 / (distance-1.0)
        #wLifted *= C/distanceWeight.sum()
        wLifted *= C/len(wLifted)

        print numpy.abs(wLocal).sum(), numpy.abs(wLifted).sum()

        # write the weighs into the objective
        obj.setLiftedEdgesCosts(wLifted, overwrite=True)
        obj.setGraphEdgesCosts(wLocal, overwrite=True)
        


        # warm start with normal multicut
        mcObj =  nifty.graph.multicut.multicutObjective(rag, wLocal)
        solverFactory = mcObj.multicutIlpCplexFactory()
        solver = solverFactory.create(mcObj)
        visitor = mcObj.multicutVerboseVisitor()
        argMc = solver.optimize(visitor)
        emc = obj.evalNodeLabels(argMc)


        # finaly optimize it
        solverFactory = obj.liftedMulticutAndresGreedyAdditiveFactory()
        solver = solverFactory.create(obj)
        visitor = obj.verboseVisitor()
        arg = solver.optimize(visitor)
        eg = obj.evalNodeLabels(arg)
        

        solverFactory = obj.liftedMulticutAndresKernighanLinFactory()
        solver = solverFactory.create(obj)
        visitor = obj.verboseVisitor()
        arg = solver.optimize(visitor, arg.copy())
        ekl = obj.evalNodeLabels(arg)

        
        print "e",emc,eg,ekl

        #projectToPixels
        pixelData = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg.astype('uint32'))
        
        vigra.segShow(rawData[sliceIndex,:,:], pixelData)
        vigra.show()
