from __future__ import print_function
from __future__ import division 

from filters import *
import fastfilters
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
pmapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_binary_full.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_5.h5"

out = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/"

ilpFile = "/home/tbeier/block_labels.ilp"


rawH5File  = h5py.File(rawFile,'r')
osegH5File = h5py.File(oversegFile,'r')
pmapH5File = h5py.File(pmapFile,'r')


blockShape = [150] * 3
#halo = [50] * 3

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


sigmas = [2.0,4.0,8.0]
maxSigma = max(sigmas)
# compute means for filters
# - get a prototypical block




filtList = []
filtListPmap = []
if False:
    bs = 100
    hs = [s//2 for s in shape]
    raw  = rawDset[hs[0] : hs[0] + bs, hs[1] : hs[1] + bs, hs[2] : hs[2] + bs].astype('float32')
    pm  = pmapDset[hs[0] : hs[0] + bs, hs[1] : hs[1] + bs, hs[2] : hs[2] + bs].astype('float32')

    for sigma in sigmas:

        filtList.append(GaussianGradientMagnitude(raw=raw, sigma=sigma))
        filtList.append(LaplacianOfGaussian(raw=raw, sigma=sigma))
        filtList.append(HessianOfGaussianEv(raw=raw, sigma=sigma))
        filtList.append(GaussianSmoothing(raw=raw, sigma=sigma))





        filtListPmap.append(GaussianGradientMagnitude(raw=pm, sigma=sigma))
        filtListPmap.append(LaplacianOfGaussian(raw=pm, sigma=sigma))
        filtListPmap.append(HessianOfGaussianEv(raw=pm, sigma=sigma))
        filtListPmap.append(GaussianSmoothing(raw=pm, sigma=sigma))





if False:


    def f(blockIndex):

        # labels halo
        blockWithBorder =  blocking.getBlockWithHalo(blockIndex, [0,0,0],[1,1,1])
        outerBlock = blockWithBorder.outerBlock
        bAcc, eAcc = outerBlock.begin, outerBlock.end


        # filters
        filterRes= []
        halo = [round(3.5*maxSigma + 3.0)] * 3
        blockWithBorder = blocking.addHalo(outerBlock, [20,20,20])
        outerBlock = blockWithBorder.outerBlock
        innerBlockLocal = blockWithBorder.innerBlockLocal
        bFilt, eFilt = outerBlock.begin, outerBlock.end
        bFiltCore, eFiltCore = innerBlockLocal.begin, innerBlockLocal.end

        with lock:
            rawFilt  = rawDset[bFilt[0]:eFilt[0], bFilt[1]:eFilt[1], bFilt[2]:eFilt[2]].astype('float32')
            pmFilt   =  pmapDset[bFilt[0]:eFilt[0], bFilt[1]:eFilt[1], bFilt[2]:eFilt[2]].astype('float32')
            oseg = osegDset[bAcc[0]:eAcc[0], bAcc[1]:eAcc[1], bAcc[2]:eAcc[2]]

        for filt in filtList:
            res = filt(rawFilt)
            res = res[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2],:]
            filterRes.append(res)

        for filt in filtListPmap:
            res = filt(pmFilt)
            res = res[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2],:]
            filterRes.append(res)

        raw = rawFilt[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2]]/255.0
        pmap = pmFilt[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2]]/255.0

        filterRes.append(raw[...,None])
        filterRes.append(pmap[...,None])

        #for vd in filterRes:
        #    print(vd.shape,vd.dtype)
        #    
        voxelData = numpy.concatenate(filterRes, axis=3)
        voxelData = numpy.require(voxelData,requirements=['C_CONTIGUOUS'],dtype='float32')

        #print(voxelData.strides)
        #last axis is continous
        assert voxelData.strides[3] == 4

        blockData = nseg.BlockData(blocking=blocking, blockIndex=blockIndex, 
            numberOfChannels=voxelData.shape[3],
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
    from sklearn.ensemble import RandomForestClassifier



    #load stuff
    f5 = h5py.File(os.path.join(out,'uvIds.h5'),'r')
    uvIds = f5['data'][:]
    f5.close()



    f5 = h5py.File(os.path.join(out,'allFeat.h5'),'r')
    allFeat = f5['data'][:]
    edgeLength  = allFeat[:,0]
    f5.close()


    mi,ma = numpy.min(allFeat,axis=0), numpy.max(allFeat, axis=0)

    print(allFeat.shape)
    

    #for i,(theMin,theMax) in enumerate(zip(mi,ma)):
    #    print(i, theMin, theMax)
    #sys.exit(0)

    #print("allFeat",allFeat.shape)


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
            assert False
        else:
            i = uvToIndex[uv]
            featTrain.append(allFeat[i,:][None,:])
            lTrain.append(l)

    featTrain = numpy.concatenate(featTrain,axis=0).astype('float32')
    labelsTrain = numpy.array(lTrain,dtype='uint32')#[:,None]

    print(featTrain.shape,labelsTrain.shape)

    # train the rf
    #rf = vigra.learning.RandomForest(treeCount = 10000,
    #    #min_split_node_size=20,
    #    #sample_classes_individually=True
    #)
    #oob = rf.learnRF(featTrain, labelsTrain)

    rf = RandomForestClassifier(n_estimators=10000, n_jobs=40)
    rf.fit(featTrain, labelsTrain)
    p1 = rf.predict_proba(allFeat)[:,1]


    #print("oob",oob)
    #with vigra.Timer("predict"):
    #    p1 = rf.predictProbabilities(allFeat)[:,1]

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
    w *= edgeLength

    obj = nifty.graph.multicut.multicutObjective(g, w)


    ilpFactory =  obj.multicutIlpFactory()
    fusionMove = obj.fusionMoveSettings(mcFactory=ilpFactory)
    factory =  obj.fusionMoveBasedFactory(fusionMove=fusionMove,
        stopIfNoImprovement=200)

    solver = ilpFactory.create(obj)

    visitor = obj.multicutVerboseVisitor()

    with vigra.Timer("optimzie"):
        ret = solver.optimize(visitor=visitor)

    ret = nifty.tools.makeDense(ret)
    


    resultH5File = h5py.File(os.path.join(out,'resultSeg4.h5'),'w')
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
                                size=nBlocks, name="Braaa")










    resultH5File.close()
    rawH5File.close() 
    osegH5File.close()
    pmapH5File.close()