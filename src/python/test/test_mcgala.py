from __future__ import print_function
import nifty
import numpy

import numpy
import vigra
import glob
import os
from functools import partial


nrag = nifty.graph.rag
ngala = nifty.graph.gala
ngraph = nifty.graph
G = nifty.graph.UndirectedGraph

def make_dataset(numberOfImages = 10, noise=1.0,shape=(100,100)):

    numpy.random.seed(42)
    imgs = []
    gts = []


    for i in range(numberOfImages):

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







def makeRag(raw):
    ew = vigra.filters.hessianOfGaussianEigenvalues(raw, 1.0)[:,:,0]
    seg, nseg = vigra.analysis.watershedsNew(ew)
    seg = seg.squeeze()
    if False:
        vigra.segShow(raw, seg)
        vigra.show()

    # get the rag
    seg -= 1
    assert seg.min() == 0
    assert seg.max() == nseg -1 
    return nifty.graph.rag.gridRag(seg)




def makeFeatureOp(rag, raw, minVal=None, maxVal=None):

    #  feature accumulators
    if minVal is None:
        minVal = float(raw.min())

    if maxVal is None:
        maxVal = float(raw.max())


    edgeFeatures = nifty.graph.rag.defaultAccEdgeMap(rag, minVal, maxVal)
    nodeFeatures = nifty.graph.rag.defaultAccNodeMap(rag, minVal, maxVal)

    # accumulate features
    nrag.gridRagAccumulateFeatures(graph=rag,data=raw,
        edgeMap=edgeFeatures, nodeMap=nodeFeatures)


    fOp = ngala.galaDefaultAccFeature(graph=rag, edgeFeatures=edgeFeatures, nodeFeatures=nodeFeatures)
    
    return fOp,minVal, maxVal

def makeEdgeGt(rag, gt):
    # get the gt
    nodeGt = nrag.gridRagAccumulateLabels(rag, gt)
    uvIds = rag.uvIds()
    edgeGt = (nodeGt[uvIds[:,0]] != nodeGt[uvIds[:,1]]).astype('double')
    return edgeGt
    


def test_mcgala():



    # get the dataset
    imgs,gts = make_dataset(2,noise=5.0,shape=(100,100))


    ragTrain  = makeRag(imgs[0])
    fOpTrain, minVal, maxVal = makeFeatureOp(ragTrain, imgs[0])
    edgeGt = makeEdgeGt(ragTrain, gts[0])


    # gala class
    settings = G.galaSettings(threshold0=0.1, threshold1=0.9, thresholdU=0.1,numberOfEpochs=1, numberOfTrees=10)
    gala = G.gala(settings)



    trainingInstance  = ngala.galaTrainingInstance(ragTrain, fOpTrain, edgeGt)
    gala.addTrainingInstance(trainingInstance)
    gala.train()


    ragTest  = makeRag(imgs[1])
    fOpTest, minVal, maxVal = makeFeatureOp(ragTest, imgs[1])
    instance  = ngala.galaInstance(ragTest, fOpTest)
    edgeGt = makeEdgeGt(ragTest, gts[1])

    nodeRes = gala.predict(instance)


    for edge, uv in enumerate(ragTest.uvIds()):
        print(edge,edgeGt[edge],nodeRes[uv[0]]!=nodeRes[uv[1]])



test_mcgala()
