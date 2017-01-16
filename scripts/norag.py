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
import nifty.graph.agglo as nagglo

from get_ilp_labels import *




def singleBlockPipeline(rawFile, pmapFile, oversegFile, ilpFile, outFolder, outName, sigmas=(2.0,4.0,8.0), filtOnRaw=True, filtOnPmap=True, blockShape=(254,)*3,
    minSegSize=None ,beta=0.5, stages=[1,1,1]):


 
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)


    # save stuff
    def saveRes(data,name):
        f5 = h5py.File(os.path.join(outFolder,'%s_%s.h5'%(outName,name)),'w')
        f5['data'] = data
        f5.close()

    # load stuff
    def loadRes(name):
        f5 = h5py.File(os.path.join(outFolder,'%s_%s.h5'%(outName,name)),'r')
        data = f5['data'][:]
        f5.close()
        return data








    rawH5File  = h5py.File(rawFile,'r')
    osegH5File = h5py.File(oversegFile,'r')
    pmapH5File = h5py.File(pmapFile,'r')


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
    lock2 = threading.Lock()
    blockRes = [None]*nBlocks


    if len(sigmas) > 0:
        maxSigma = max(sigmas)
    



    if stages[0] == 1:
        print("STAGE 0")
        filtList = []
        filtListPmap = []

        bs = 100
        hs = [s//2 for s in shape]
        raw  = rawDset[hs[0] : hs[0] + bs, hs[1] : hs[1] + bs, hs[2] : hs[2] + bs].astype('float32')
        pm  =  pmapDset[hs[0] : hs[0] + bs, hs[1] : hs[1] + bs, hs[2] : hs[2] + bs].astype('float32')

        for sigma in sigmas:

            if filtOnRaw:
                filtList.append(GaussianGradientMagnitude(raw=raw, sigma=sigma))
                filtList.append(LaplacianOfGaussian(raw=raw, sigma=sigma))
                filtList.append(HessianOfGaussianEv(raw=raw, sigma=sigma))
                filtList.append(GaussianSmoothing(raw=raw, sigma=sigma))

            if filtOnPmap:
                filtListPmap.append(GaussianGradientMagnitude(raw=pm, sigma=sigma))
                filtListPmap.append(LaplacianOfGaussian(raw=pm, sigma=sigma))
                filtListPmap.append(HessianOfGaussianEv(raw=pm, sigma=sigma))
                filtListPmap.append(GaussianSmoothing(raw=pm, sigma=sigma))




        def f(blockIndex):

            # labels halo
            blockWithBorder =  blocking.getBlockWithHalo(blockIndex, [0,0,0],[1,1,1])
            outerBlock = blockWithBorder.outerBlock
            bAcc, eAcc = outerBlock.begin, outerBlock.end


            # filters
            
            filterRes= []

            if len(sigmas) > 0 and (filtOnRaw or filtOnPmap):
                halo = [int(round(3.5*maxSigma + 3.0))] * 3
            else:
                halo = [0]*3

            blockWithBorder = blocking.addHalo(outerBlock, halo)
            outerBlock = blockWithBorder.outerBlock
            innerBlockLocal = blockWithBorder.innerBlockLocal
            bFilt, eFilt = outerBlock.begin, outerBlock.end
            bFiltCore, eFiltCore = innerBlockLocal.begin, innerBlockLocal.end

            with lock:
                rawFilt  = rawDset[bFilt[0]:eFilt[0], bFilt[1]:eFilt[1], bFilt[2]:eFilt[2]].astype('float32')
                pmFilt   = pmapDset[bFilt[0]:eFilt[0], bFilt[1]:eFilt[1], bFilt[2]:eFilt[2]].astype('float32')
                oseg     = osegDset[bAcc[0]:eAcc[0], bAcc[1]:eAcc[1], bAcc[2]:eAcc[2]]

            raw = rawFilt[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2]]/255.0
            pmap = pmFilt[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2]]/255.0


            filterRes.append(raw[...,None])
            filterRes.append(pmap[...,None])

            for filt in filtList:
                res = filt(rawFilt)
                res = res[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2],:]
                filterRes.append(res)

            for filt in filtListPmap:
                res = filt(pmFilt)
                res = res[bFiltCore[0]:eFiltCore[0], bFiltCore[1]:eFiltCore[1], bFiltCore[2]:eFiltCore[2],:]
                filterRes.append(res)




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

        nodeCounts   = mergedRes.nodeCounts()
        edgeCounts   = mergedRes.edgeCounts()

        # mean pmap
        nodeMeansP   = mergedRes.nodeMeans(1)
        edgeMeansP   = mergedRes.edgeMeans(1)



        saveRes(nodeCounts,'nodeCounts')
        saveRes(edgeCounts,'edgeCounts')
        saveRes(nodeMeansP,'nodeMeansP')
        saveRes(edgeMeansP,'edgeMeansP')
        saveRes(uvIds,'uvIds')
        saveRes(allFeat,'allFeat')
            

    if stages[1] == 1:

        from sklearn.ensemble import RandomForestClassifier



        #load stuff
        uvIds = loadRes('uvIds')
        allFeat = loadRes('allFeat')

        nodeCounts = loadRes('nodeCounts')
        edgeCounts = loadRes('edgeCounts')
        nodeMeansP = loadRes('nodeMeansP')
        edgeMeansP = loadRes('edgeMeansP')


        # get the graph
        numberOfNodes = uvIds.max() + 1
        g = nifty.graph.UndirectedGraph(numberOfNodes)
        g.insertEdges(uvIds)

        print("graph",g)

        if False:
            print("ucm feat")
            print("edgeMeansP",edgeMeansP)
            ucmFeat = nagglo.ucmFeatures(graph=g, edgeIndicators=edgeMeansP, 
                               edgeSizes=edgeCounts, nodeSizes=nodeCounts,
                               sizeRegularizers=numpy.arange(0.025,1,0.1))

            allFeat = numpy.concatenate([allFeat, ucmFeat],axis=1)
            sys.exit()

        print("rest")
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
                print(g,uv)
                assert False
            else:
                i = uvToIndex[uv]
                featTrain.append(allFeat[i,:][None,:])
                lTrain.append(l)

        featTrain = numpy.concatenate(featTrain,axis=0).astype('float32')
        labelsTrain = numpy.array(lTrain,dtype='uint32')#[:,None]

        print("training set size",featTrain.shape,labelsTrain.shape)

        # train the rf
        #rf = vigra.learning.RandomForest(treeCount = 10000,
        #    #min_split_node_size=20,
        #    #sample_classes_individually=True
        #)
        #oob = rf.learnRF(featTrain, labelsTrain)

        rf = RandomForestClassifier(n_estimators=1000, n_jobs=40)
        rf.fit(featTrain, labelsTrain)
        p1 = rf.predict_proba(allFeat)[:,1]


        #print("oob",oob)
        #with vigra.Timer("predict"):
        #    p1 = rf.predictProbabilities(allFeat)[:,1]



        # get weights
        eps = 0.00001
        beta = float(beta)

        p1 = numpy.clip(p1, eps, 1.0 - eps)
        p0 = 1.0 - p1
        w = numpy.log(p0/p1) + numpy.log((1.0-beta)/beta)
        w *= edgeCounts

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
        

        resName = '%s_%s.h5'%(outName,"resultSeg")
        resultH5File = h5py.File(os.path.join(outFolder,resName),'w')
        resultSegDset = resultH5File.create_dataset('data', shape=shape, chunks=(64,)*3, compression='gzip', dtype='uint32')



        blocking = nifty.tools.blocking(roiBegin=[0]*3, roiEnd=shape, 
                                    blockShape=(256,)*3)
        nBlocks = blocking.numberOfBlocks

        def f(blockIndex):

            # labels halo
            block =  blocking.getBlock(blockIndex)
            b, e = block.begin, block.end

            # load files
            with lock:
                oseg = osegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

            relabeld = numpy.take(ret, oseg)

            with lock2:
                resultSegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = relabeld

            #print("relabeld shape",relabeld.shape)





        nifty.tools.parallelForEach(iterable=range(nBlocks), f=f, 
                                    nWorkers=-1, showBar=True,
                                    size=nBlocks, name="Braaa")
        resultH5File.close()
        

    if stages[2] == 1:

        if minSegSize is not None:
            with vigra.Timer("load the segIn"):
                resName = '%s_%s.h5'%(outName,"resultSeg")
                resultH5File = h5py.File(os.path.join(outFolder,resName),'r')
                segIn = resultH5File['data'][:,:,:]#[0:200,0:200,0:200]

            with vigra.Timer("load the pmap"):
                # load the pmap
                pmap = pmapDset[:,:,:,]#[0:200,0:200,0:200]

            with vigra.Timer("make rag"):
                rag = nifty.graph.rag.gridRag(segIn)
                print(rag)
            
            with vigra.Timer("acc"):
                eFeatures, nFeatures = nifty.graph.rag.accumulateMeanAndLength(rag, pmap, [100,100,100],-1)
                eMeans = eFeatures[:,0]
                eSizes = eFeatures[:,1]
                nSizes = nFeatures[:,1]

            with vigra.Timer("cluster"):
                res = nagglo.sizeLimitClustering(graph=rag, nodeSizes=nSizes, minimumNodeSize=int(minSegSize), 
                                          edgeIndicators=eMeans,edgeSizes=eSizes, 
                                          sizeRegularizer=0.001, gamma=0.999,
                                          makeDenseLabels=True)
                #res = numpy.arange(rag.nodeIdUpperBound + 1)

            resName = '%s_%s.h5'%(outName,"outSegSmallRemoved")
            resultH5File = h5py.File(os.path.join(outFolder,resName),'w')
            resultSegDset = resultH5File.create_dataset('data', shape=shape, chunks=(64,)*3, compression='gzip', dtype='uint32')


            with vigra.Timer("makeDense"):
                def f(blockIndex):

                    # labels halo
                    block =  blocking.getBlock(blockIndex)
                    b, e = block.begin, block.end


                    oseg = segIn[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

                    relabeld = numpy.take(res, oseg)

                    with lock:
                        resultSegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]] = relabeld

                    #print("relabeld shape",relabeld.shape)





                nifty.tools.parallelForEach(iterable=range(nBlocks), f=f, 
                                            nWorkers=-1, showBar=True,
                                            size=nBlocks, name="write results")


    
    rawH5File.close() 
    pmapH5File.close()
    osegH5File.close()

# 2nm
if False:
    rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_2nm.h5"
    pmapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_binary_full.h5"


    if False:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_5.h5"
        ilpFile = "/home/tbeier/block_labels.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/"


    if False:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/smallSegRemoved.h5"
        #oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/resultSeg5.h5"
        ilpFile = "/home/tbeier/grande.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2kr2/" 



        singleBlockPipeline(rawFile=rawFile, pmapFile=pmapFile, 
                            oversegFile=oversegFile, ilpFile=ilpFile,
                            outFolder=outFolder, outName="resr2", 
                            sigmas=(2.0,4.0,8.0),
                            filtOnRaw=False, filtOnPmap=False, 
                            blockShape=(254,)*3,
                            minSegSize=None ,beta=0.75, stages=[0,1,0])

# 4nm
if True:

    rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_4nm.h5"
    pmapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_binary_4nm.h5"


    if False:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel4nm/_4nm_agglo_seg0.3_20.h5"
        #oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/resultSeg5.h5"
        ilpFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel4nm/labels.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt4nm/" 



        singleBlockPipeline(rawFile=rawFile, pmapFile=pmapFile, 
                            oversegFile=oversegFile, ilpFile=ilpFile,
                            outFolder=outFolder, outName="res4nm", 
                            sigmas=(1.0,2.0,4.0),
                            filtOnRaw=True, filtOnPmap=True, 
                            blockShape=(125,)*3,
                            minSegSize=15**3 ,beta=0.75, stages=[0,0,1])


    if True:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt4nm/res4nm_outSegSmallRemoved.h5"
        #oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/resultSeg5.h5"
        ilpFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel4nm/labelsr2.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt4nmr2/" 



        singleBlockPipeline(rawFile=rawFile, pmapFile=pmapFile, 
                            oversegFile=oversegFile, ilpFile=ilpFile,
                            outFolder=outFolder, outName="res4nmr2", 
                            sigmas=(1.0,2.0,4.0),
                            filtOnRaw=False, filtOnPmap=False, 
                            blockShape=(125,)*3,
                            minSegSize=15**3 ,beta=0.75, stages=[0,1,0])


# 8nm
if False:

    rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_8nm.h5"
    pmapFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/pc_out/2nm/prediction_binary_8nm.h5"


    if False:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel8nm/_8nm_agglo_seg0.3_40.h5"
        #oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/resultSeg5.h5"
        ilpFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel8nm/labels.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt8nm/" 



        singleBlockPipeline(rawFile=rawFile, pmapFile=pmapFile, 
                            oversegFile=oversegFile, ilpFile=ilpFile,
                            outFolder=outFolder, outName="res8nm", 
                            sigmas=(0.75,2.0,3.0),
                            filtOnRaw=True, filtOnPmap=True, 
                            blockShape=(75,)*3,
                            minSegSize=7**3 ,beta=0.75, stages=[0,1,1])


    if True:
        oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt8nm/res8nm_outSegSmallRemoved.h5"
        #oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt2k/resultSeg5.h5"
        ilpFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxel8nm/labelsr2.ilp"
        outFolder = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/opt8nmr2/" 



        singleBlockPipeline(rawFile=rawFile, pmapFile=pmapFile, 
                            oversegFile=oversegFile, ilpFile=ilpFile,
                            outFolder=outFolder, outName="res8nmr2", 
                            sigmas=(0.75,2.0,3.0),
                            filtOnRaw=False, filtOnPmap=False, 
                            blockShape=(75,)*3,
                            minSegSize=15**3 ,beta=0.75, stages=[1,1,0])