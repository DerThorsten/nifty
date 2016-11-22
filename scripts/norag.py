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




rawH5File  = h5py.File(os.path.join(out,'raw.h5'))
osegH5File = h5py.File(os.path.join(out,'oseg.h5'))
pmapH5File = h5py.File(os.path.join(out,'pmap.h5'))

ilpFile = os.path.join(out,'edge_labels.ilp')


blockShape = [200] * 3
halo = [50] * 3

# dsets
rawDset  = rawH5File['data']  
osegDset = osegH5File['data'] 
pmapDset = pmapH5File['data'] 

shape = rawDset.shape

#load labels
uv, labels = getIlpLabels(ilpFile)

# blocking
blocking = nifty.tools.blocking(roiBegin=[0]*3, roiEnd=shape, 
                                blockShape=blockShape)
nBlocks = blocking.numberOfBlocks


lock = threading.Lock()


class Foo(object):
    def merge(self, other):
        pass




blockRes = [None]*nBlocks




def f(blockIndex):

    # labels halo
    blockWithBorder =  blocking.getBlockWithHalo(blockIndex, [0,0,0],[1,1,1])
    outerBlock = blockWithBorder.outerBlock
    b, e = outerBlock.begin, outerBlock.end

    # load files
    with lock:
        oseg = osegDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
        #raw  = rawDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]
        #pmap = pmapDset[b[0]:e[0], b[1]:e[1], b[2]:e[2]]

    blockData = nseg.BlockData(blocking, blockIndex)
    blockData.accumulate(oseg)

    blockRes[blockIndex] = blockData






nifty.tools.parallelForEach(iterable=range(nBlocks), f=f, 
                            nWorkers=-1, showBar=True,
                            size=nBlocks, name="ComputeFeatures")


def mergePairwise(toMerge):

    while(len(toMerge) != 1):
        newList = []

        l = len(toMerge)
        pairs  = l//2
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
                newList.append(blocKRes[p0])

        toMerge = newList
    return toMerge[0]

mergedRes = mergePairwise(blockRes) 








rawH5File.close() 
osegH5File.close()
pmapH5File.close()