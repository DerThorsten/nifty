from __future__ import print_function
from __future__ import division

import nifty
import nifty.viewer
import numpy
import vigra
import h5py 
import os

import threading


# needed files: raw, pmap, supervoxel
rawFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/datasets/hhess_2nm/raw_2k_2nm.h5"
pmapFile = "/home/tbeier/prediction_semantic_binary_full.h5"
oversegFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/supervoxels/2nm/aggloseg_0.3_50.h5"


blockFile = "/media/tbeier/4cf81285-be72-45f5-8c63-fb8e9ff4476c/blocks"

blockCoreShape = (500, 500, 500)
halo = (100,100,100)

totalShape = (2000,2000,2000)
blocking = nifty.tools.blocking(roiBegin=(0,0,0), roiEnd=totalShape, blockShape=blockCoreShape)


rawFileH5     = h5py.File(rawFile,'r')
pmapFileH5    = h5py.File(pmapFile,'r')
oversegFileH5 = h5py.File(oversegFile,'r')

rawFileData = rawFileH5['data']
pmapFileData = pmapFileH5['data']
oversegFileData = oversegFileH5['data']




lock = threading.Lock()
done = [0]


with nifty.tools.progressBar(size=blocking.numberOfBlocks) as bar:

    def f(blockIndex):




        blockWithHalo = blocking.getBlockWithHalo(blockIndex, list(halo))
        outerBlock = blockWithHalo.outerBlock
        b,e = outerBlock.begin, outerBlock.end
        shape = outerBlock.shape

        blockRawFile = os.path.join(blockFile,"block_%d_raw.h5"%blockIndex)
        blockPmapFile = os.path.join(blockFile,"block_%d_pmap.h5"%blockIndex)
        blockOversegFile = os.path.join(blockFile,"block_%d_oseg.h5"%blockIndex)

        blockRawFileH5 = h5py.File(blockRawFile,'w')
        blockPmapFileH5 = h5py.File(blockPmapFile,'w')
        blockOversegFileH5 = h5py.File(blockOversegFile,'w')


        blockRawFileData = rawFileData[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
        blockPmapFileData = pmapFileData[b[0]:e[0], b[1]:e[1], b[2]:e[2],:][:,:,:,0]
        blockOversegFileData = oversegFileData[b[0]:e[0], b[1]:e[1], b[2]:e[2]] 

        blockRawFileH5.create_dataset('data',shape=shape, chunks=(100,100,100),data=blockRawFileData,dtype='uint8')
        blockPmapFileH5.create_dataset('data',shape=shape, chunks=(100,100,100),data=blockPmapFileData,dtype='float32')
        blockOversegFileH5.create_dataset('data',shape=shape, chunks=(100,100,100),data=blockOversegFileData,dtype='uint32',compression="gzip")


        blockRawFileH5.close()
        blockPmapFileH5.close()
        blockOversegFileH5.close()
            
        with lock:
            done[0] += 1
            bar.update(done[0])



    nifty.tools.parallelForEach(range(blocking.numberOfBlocks), f=f, nWorkers=1)





rawFileH5.close()   
pmapFileH5.close()
oversegFileH5.close()