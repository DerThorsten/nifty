from __future__ import print_function
from __future__ import division 

import numpy
import h5py
import os
import vigra
import nifty
import threading
import nifty.pipelines.neuro_seg as nseg

from get_ilp_labels import *

out = "/home/tbeier/Desktop/play"




rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_2nm.h5"
pmapFile = "/home/tbeier/prediction_semantic_binary_full.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_50.h5"

out = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/"

ilpFile = "/home/tbeier/block_labels.ilp"


rawH5File  = h5py.File(rawFile)
osegH5File = h5py.File(oversegFile)
pmapH5File = h5py.File(pmapFile)


blockShape = [200] * 3
halo = [50] * 3

# dsets
rawDset  = rawH5File['data']  
osegDset = osegH5File['data'] 
pmapDset = pmapH5File['data'] 

print("raw",rawDset.shape)
print("oseg",osegDset.shape)
print("pmap",pmapDset.shape)

shape = rawDset.shape

#load labels
uvIdsTrain, labelsTrain = getIlpLabels(ilpFile)


blocking = nifty.tools.blocking(roiBegin=[0]*3, roiEnd=shape, 
                                blockShape=blockShape)
nBlocks = blocking.numberOfBlocks
lock = threading.Lock()
blockRes = [None]*nBlocks

if True:


    def f(blockIndex):

        # labels halo
        blockWithBorder =  blocking.getBlockWithHalo(blockIndex, [0,0,0],[1,1,1])
        outerBlock = blockWithBorder.outerBlock
        b, e = outerBlock.begin, outerBlock.end

        # load files
        with lock:
            oseg = osegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

            # for the features
            raw  = rawDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]].astype('float32') / 255.0
            pmap = pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2],:][...,0].astype('float32') /255.0

        voxelData = numpy.concatenate([raw[...,None], pmap[:,:,:,None]],axis=3)

        #last axis is continous
        assert voxelData.strides[3] == 4

        blockData = nseg.BlockData(blocking=blocking, blockIndex=blockIndex, 
            numberOfChannels=2,
            numberOfBins=20)

        blockData.accumulate(oseg, voxelData)

        blockRes[blockIndex] = blockData






    nifty.tools.parallelForEach(iterable=range(nBlocks), f=f, 
                                nWorkers=-1, showBar=True,
                                size=nBlocks, name="ComputeFeatures")


    def mergePairwise(toMerge):

        while(len(toMerge) != 1):
            newList = []

            l = len(toMerge)
            pairs  = (l+1)//2
            print(l)
            for p in range(pairs):
                p0 = p*2
                p1 = p*2 + 1

                if(p1<l):
                    b0 = toMerge[p0]
                    b1 = toMerge[p1]
                    b0.merge(b1)
                    newList.append(b0)
                else:
                    newList.append(toMerge[p0])

            toMerge = newList
        return toMerge[0]

    mergedRes = mergePairwise(blockRes) 



    uvIds = mergedRes.uvIds()
    allFeat = mergedRes.extractFeatures()


    # save stuff
    
    f5 = h5py.File(os.path.join(out,'uvIds.h5'),'w')
    f5['data'] = uvIds
    f5.close()


    f5 = h5py.File(os.path.join(out,'allFeat.h5'),'w')
    f5['data'] = allFeat
    f5.close()

if True:

    #load stuff
    f5 = h5py.File(os.path.join(out,'uvIds.h5'),'r')
    uvIds = f5['data'][:]
    f5.close()



    f5 = h5py.File(os.path.join(out,'allFeat.h5'),'r')
    allFeat = f5['data'][:]
    f5.close()


    # mapping
    uvToIndex = dict()
    for i,uv in enumerate(uvIds):   
        uv = long(uv[0]),long(uv[1])
        uvToIndex[uv] = i

    featTrain = []
    lTrain = []

    for i,(uv,l) in enumerate(zip(uvIdsTrain,labelsTrain)):   
        uv = long(uv[0]),long(uv[1])
        if uv not in uvToIndex:
            print("WHY?!?")
        else:
            i = uvToIndex[uv]
            featTrain.append(allFeat[i,:][None,:])
            lTrain.append(l)

    featTrain = numpy.concatenate(featTrain,axis=0).astype('float32')
    labelsTrain = numpy.array(lTrain,dtype='uint32')[:,None]

    print(featTrain.shape,labelsTrain.shape)

    # train the rf
    rf = vigra.learning.RandomForest(treeCount = 255)
    oob = rf.learnRF(featTrain, labelsTrain)

    print("oob",oob)

    # do the predictions
    p1 = rf.predictProbabilities(allFeat)[:,1]

    # get the graph
    numberOfNodes = uvIds.max() + 1
    g = nifty.graph.UndirectedGraph(numberOfNodes)
    g.insertEdges(uvIds)

    # get weights
    eps = 0.00001
    beta = 0.5

    p1 = numpy.clip(p1, eps, 1.0 - eps)
    p0 = 1.0 - p1
    w = numpy.log(p0/p1) + numpy.log((1.0-beta)/beta)



    obj = nifty.graph.multicut.multicutObjective(g, w)


    ilpFactory =  obj.multicutIlpFactory()
    fusionMove = obj.fusionMoveSettings(mcFactory=ilpFactory)
    factory =  obj.fusionMoveBasedFactory(fusionMove=fusionMove,
        stopIfNoImprovement=200)

    solver = ilpFactory.create(obj)

    visitor = obj.multicutVerboseVisitor()
    ret = solver.optimize(visitor=visitor)

    ret = nifty.tools.makeDense(ret)
    


    resultH5File = h5py.File(os.path.join(out,'resultSeg3.h5'),'w')
    resultSegDset = resultH5File.create_dataset('data', shape=shape, chunks=(64,)*3, compression='gzip', dtype='uint32')



    def f(blockIndex):

        # labels halo
        block =  blocking.getBlock(blockIndex)
        b, e = block.begin, block.end

        # load files
        with lock:
            oseg = osegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

        relabeld = numpy.take(ret, oseg)

        with lock:
            resultSegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = relabeld

        #print("relabeld shape",relabeld.shape)





    nifty.tools.parallelForEach(iterable=range(nBlocks), f=f, 
                                nWorkers=-1, showBar=True,
                                size=nBlocks, name="ComputeFeatures")










    resultH5File.close()
    rawH5File.close() 
    osegH5File.close()
    pmapH5File.close()