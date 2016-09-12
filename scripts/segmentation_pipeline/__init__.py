import vigra
import nifty
import numpy
import random
import os
import h5py

from tools import *
from superpixels import *
from rag import *



def runPipeline(trainInput, testInput, settings):
    


    random.seed(42)

    # prepare data
    trainRaw = numpy.array(vigra.impex.readVolume(trainInput['rawData'])).squeeze()
    trainGt  = numpy.array(vigra.impex.readVolume(trainInput['gt'])).squeeze()
    testRaw  = numpy.array(vigra.impex.readVolume(testInput['rawData'])).squeeze()

    trainRaw = numpy.rollaxis(trainRaw, 2, 0 )[0:6,:,:]
    testRaw = numpy.rollaxis(testRaw, 2, 0 )  [0:6,:,:]


    # prepare output folders
    rootOutDir = settings['rootOutDir']




    trainTestDict = {
       'train' : {
           'input' : trainInput,
           'raw'   : trainRaw,
           'gt'    : trainGt
        }
        ,
        'test' : {
            'input' : testInput,
            'raw'   : testRaw
        }
    }

    # create outfolders
    for key in   trainTestDict.keys():
        outDir = os.path.join(rootOutDir, key)
        print outDir
        ensureDir(outDir)
        trainTestDict[key]['outDir'] = outDir


    # init  superpixels 
    # either load them or compute them
    # in both cases they are stored at a common place
    for key in trainTestDict:
        computeSuperpixels(trainTestDict[key], settings)


    # init region adjacency graph
    for key in trainTestDict:
        computeRag(trainTestDict[key], settings)

    # compute rag features
    for key in trainTestDict:
        computeRagFeatures(trainTestDict[key], settings)




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


def getSuperpixels(dataDict, settings):
    outDir = dataDict['outDir']
    if 'superpixels' in dataDict:
        return dataDict['superpixels']
    else:
        spFile = os.path.join(outDir,'supervoxels.h5')
        sp =  h5Read(spFile)
        dataDict['superpixels'] = sp
        return  sp


def computeRag(dataDict, settings):

    rawData = dataDict['raw']
    superpixels = getSuperpixels(dataDict, settings)
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
    ragsAndSuperpixels = getRagsAndSuperpixels(dataDict, settings)

    for sliceIndex  in range(rawData.shape[0]):
        rag,sp = ragsAndSuperpixels[sliceIndex]

        print rag.numberOfNodes, rag.numberOfEdges

        localRagFeatures(raw=rawData[sliceIndex,:,:], pmap=None,
                        overseg=sp, rag=rag, settings=settings)