from __future__ import print_function
from __future__ import division 

import os
import h5py
import threading

import numpy
import vigra

import nifty
import nifty.hdf5
import nifty.graph.rag



blockLabelsIlpFile = "/home/tbeier/block_labels.ilp"

rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_2nm.h5"
pmapFile = "/home/tbeier/prediction_semantic_binary_full.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_50.h5"
oversegFile = "/home/tbeier/Desktop/aggloseg_0.3_50.h5"
workDir = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k" 

def getLabels(blockLabelsIlpFile):

    allLabels = dict()
    blockLabelsIlpFileH5 = h5py.File(blockLabelsIlpFile,'r')
    edgeLabels = blockLabelsIlpFileH5['Training and Multicut']['EdgeLabelsDict']

    for key in edgeLabels.keys():
        el = edgeLabels[key]
        labels = el['labels'][:]
        spIds = el['sp_ids'][:,:]

        for l, spId in zip(labels, spIds):
            if l == 1 or l == 2:
                uv = long(spId[0]),long(spId[1])
                if uv in allLabels:
                    allLabels[uv] = max(l, allLabels[uv])
                else:
                    allLabels[uv] = l
    blockLabelsIlpFileH5.close()


    l  = numpy.array(allLabels.values())
    uv = numpy.array(allLabels.keys())
    return l-1, uv



def h5Max(dset, blockShape=None):

    shape = dset.shape
    ndim = len(shape)
    assert ndim == 3
    if blockShape is None:
        blockShape = [100]*len(shape)

    blocking = nifty.tools.blocking(roiBegin=(0,0,0), roiEnd=shape, blockShape=list(blockShape))


    lock = threading.Lock()
    done = [0]
    gMax = [0] 

    with nifty.tools.progressBar(size=blocking.numberOfBlocks) as bar:

        def f(blockIndex):




            block = blocking.getBlock(blockIndex)
            b,e = block.begin, block.end
            

            d = dset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

            with lock:
                done[0] += 1
                bar.update(done[0])

                gMax[0] = max(d.max(),gMax[0])

        nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=1)

    return gMax[0]



def computeRag(oversegFile, dset):

    oversegFileH5 = h5py.File(oversegFile)
    oversegDset = oversegFileH5[dset]
    try:
        maxLabel = oversegDset.attrs['maxLabel']
        #print("load max label")
    except:
        print("compute max label")
        maxLabel = h5Max(oversegDset)
        oversegDset.attrs['maxLabel'] = maxLabel

    oversegFileH5.close()



    print("max label",maxLabel)


    nLabels = maxLabel + 1

    cs = nifty.hdf5.CacheSettings()
    cs.hashTabelSize = 977
    cs.nBytes = 990000000
    cs.rddc = 0.5

    h5File = nifty.hdf5.openFile(oversegFile, cs)
    labelsArray = nifty.hdf5.Hdf5ArrayUInt32(h5File, dset)

    ragFile = os.path.join(workDir,'rag.h5')
    if not os.path.isfile(ragFile):
        
        with vigra.Timer("rag"):
            gridRag = nifty.graph.rag.gridRagHdf5(labelsArray, nLabels, blockShape=[150,150,150], numberOfThreads=20)

        print("serialize")
        serialization = gridRag.serialize()
        
        print("save serialization")
        ragH5File = h5py.File(ragFile)
        ragH5File['data'] = serialization
        ragH5File.close()
    else:
        ragH5File = h5py.File(ragFile,'r')
        serialization = ragH5File['data']
        gridRag = nifty.graph.rag.gridRagHdf5(labelsArray, nLabels, serialization=serialization)
        ragH5File.close()


    return gridRag


uv,l = getLabels(blockLabelsIlpFile)
oversegFileH5 = h5py.File(oversegFile)



# rag
rag = computeRag(oversegFile,'data')



h5File = nifty.hdf5.openFile(rawFile,)
rawArray = nifty.hdf5.Hdf5ArrayUInt8(h5File, 'data')


# accumulate features
rawFeat = nifty.graph.rag.accumulateStandartFeatures(rag=rag, data=rawArray, minVal=0.0, maxVal=255.0, blockShape=[75,75,75], numberOfThreads=-1)






