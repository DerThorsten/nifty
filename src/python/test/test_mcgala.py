from __future__ import print_function
import nifty
import numpy

import numpy
import vigra
from progressbar import *
import glob
import os
from functools import partial

def make_dataset(numberOfImages = 10):

    numpy.random.seed(42)
    nImages = numberOfImages 
    shape = [100, 100]
    noise = 1.0
    imgs = []
    gts = []


    for i in range(nImages):

        gtImg = numpy.zeros(shape)
        gtImg[0:shape[0]/2,:] = 1

        gtImg[shape[0]/4: 3*shape[0]/4, shape[0]/4: 3*shape[0]/4]  = 2

        ra = numpy.random.randint(180)
        #print ra 

        gtImg = vigra.sampling.rotateImageDegree(gtImg.astype(numpy.float32),int(ra),splineOrder=0)

        #if i<1 :
        #    vigra.imshow(gtImg)
        #    vigra.show()

        img = gtImg + numpy.random.random(shape)*float(noise)
        #if i<1 :
        #    vigra.imshow(img)
        #    vigra.show()

        imgs.append(img.astype('float32'))
        gts.append(gtImg)


    return imgs,gts















def test_mcgala():
    nrag = nifty.graph.rag
    ngala = nifty.graph.gala
    ngraph = nifty.graph

    # get the dataset
    imgs,gts = make_dataset(2)


    # get the supervoxels
    rawTrain  =imgs[0]
    ew = vigra.filters.hessianOfGaussianEigenvalues(rawTrain, 1.0)[:,:,0]
    seg, nseg = vigra.analysis.watershedsNew(ew)
    seg = seg.squeeze()
    if False:
        vigra.segShow(rawTrain, seg)
        vigra.show()

    # get the rag
    seg -= 1
    assert seg.min() == 0
    assert seg.max() == nseg -1 
    ragTrain = nifty.graph.rag.gridRag(seg)
    print("#nodes",ragTrain.numberOfNodes,"#edges",ragTrain.numberOfEdges)



    #  feature accumulators
    minVal = float(rawTrain.min())
    maxVal = float(rawTrain.max())
    edgeFeatures = nifty.graph.rag.defaultAccEdgeMap(ragTrain, minVal, maxVal)
    nodeFeatures = nifty.graph.rag.defaultAccNodeMap(ragTrain, minVal, maxVal)

    # accumulate features
    nrag.gridRagAccumulateFeatures(graph=ragTrain,data=rawTrain,
        edgeMap=edgeFeatures, nodeMap=nodeFeatures)


    # gala class
    gala = nifty.graph.UndirectedGraph.gala()

    # use dummy features atm
    f = ngala.GalaDummyFeatureUndirectedGraph()
    edgeGt = numpy.zeros(ragTrain.numberOfEdges,dtype='uint8')

    trainingInstance  = ngala.galaTrainingInstance(ragTrain, f, edgeGt)
    gala.addInstance(trainingInstance)
    
    #gala.addInstance(ragTrain, edgeGt)

test_mcgala()
