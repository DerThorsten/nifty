import vigra
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.agglo
import numpy
import h5py

from reraise import *

@reraise_with_stack
def makeRag(overseg, ragFile, settings):

    assert overseg.min() == 0 
    rag = nifty.graph.rag.gridRag(overseg)

    # serialize the rag
    serialization = rag.serialize()

    # save serialization
    f5 = h5py.File(ragFile, 'w') 
    f5['rag'] = serialization

    # save superpixels 
    f5['superpixels'] = overseg

    f5.close()




def loadRag(ragFile):
    # read serialization
    f5 = h5py.File(ragFile, 'r') 
    serialization = f5['rag'][:]

    # load superpixels 
    overseg = f5['superpixels'][:]

    f5.close()

    rag = nifty.graph.rag.gridRag(overseg, serialization=serialization)
    return rag, overseg




@reraise_with_stack
def localRagFeatures(raw, pmap, overseg, rag, settings):
        
    print "bincoutn", numpy.bincount(overseg.reshape([-1])).size,"nNodes",rag.numberOfNodes

    edgeFeat, nodeFeat = nifty.graph.rag.accumulateStandartFeatures(
        rag=rag,data=raw.astype('float32'),
        minVal=0.0,
        maxVal=255.0,
        blockShape=[75, 75],
        numberOfThreads=10
    )

