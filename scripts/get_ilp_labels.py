from __future__ import print_function
from __future__ import division 

import h5py
import numpy



def getIlpLabels(blockLabelsIlpFile):

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
    return uv,l-1