import vigra
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.agglo
import nifty.ground_truth
import numpy
import h5py

from reraise import *



@reraise_with_stack
def makeGt(rawData, binaryGt, settings):


    seeds = vigra.analysis.labelImageWithBackground(binaryGt)

    edgeIndicator = vigra.filters.hessianOfGaussianEigenvalues(rawData, 3.0)[:,:,0]
    seg, nseg = vigra.analysis.watershedsNew(edgeIndicator, seeds=seeds)
   

    return seg




def getTrainingData(rag, sp, pixelGt, features, settings):

    thresholds = settings['fuztGtThreshold']
    t0,t1 = thresholds

    # compute overlap
    overlap = nifty.ground_truth.Overlap(rag.numberOfNodes-1, 
        sp, pixelGt
    )

    fuzzyEdgeGt = overlap.differentOverlaps(rag.uvIds())


    where0 = numpy.where(fuzzyEdgeGt<t0)
    where1 = numpy.where(fuzzyEdgeGt>t1)

    f0 = features[where0]
    f1 = features[where1]

    feat = numpy.concatenate([f0,f1],axis=0)
    labels = numpy.ones(feat.shape[0],dtype='uint32')
    labels[0:f0.shape[0]] = 0

    return feat,labels


def getLiftedTrainingData(rag, sp, pixelGt, liftedUvIds, features, settings):

    thresholds = settings['fuztGtThreshold']
    t0,t1 = thresholds

    # compute overlap
    overlap = nifty.ground_truth.Overlap(rag.numberOfNodes-1, 
        sp, pixelGt
    )

    fuzzyEdgeGt = overlap.differentOverlaps(liftedUvIds)


    where0 = numpy.where(fuzzyEdgeGt<t0)
    where1 = numpy.where(fuzzyEdgeGt>t1)

    f0 = features[where0]
    f1 = features[where1]

    feat = numpy.concatenate([f0,f1],axis=0)
    labels = numpy.ones(feat.shape[0],dtype='uint32')
    labels[0:f0.shape[0]] = 0

    return feat,labels