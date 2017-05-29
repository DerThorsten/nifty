import vigra
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.agglo
import numpy



def makeSmallerSeg(oseg, edgeIndicator, reduceBy, sizeRegularizer):
    import nifty
    nrag = nifty.graph.rag
    nagglo = nifty.graph.agglo

    # "overseg in c order starting at zero"
    oseg = numpy.require(oseg, dtype='uint32',requirements='C')
    oseg -= 1

    # "make rag"
    rag = nifty.graph.rag.gridRag(oseg)

    # "volfeatshape"
    vFeat = numpy.require(edgeIndicator, dtype='float32',requirements='C')

    # "accumulate means and counts"
    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, vFeat, [100,100],1)

    eMeans = eFeatures[:,0]
    eSizes = eFeatures[:,1]
    nSizes = nFeatures[:,1]

    # "get clusterPolicy"

    numberOfNodesStop = int(float(rag.numberOfNodes)/float(reduceBy) + 0.5)
    numberOfNodesStop = max(1,numberOfNodesStop)
    numberOfNodesStop = min(rag.numberOfNodes, numberOfNodesStop)

    clusterPolicy = nagglo.edgeWeightedClusterPolicy(
        graph=rag, edgeIndicators=eMeans,
        edgeSizes=eSizes, nodeSizes=nSizes,
        numberOfNodesStop=numberOfNodesStop,
        sizeRegularizer=float(sizeRegularizer))

    # "do clustering"
    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run()
    seg = agglomerativeClustering.result()#out=[1,2,3,4])

    # "make seg dense"
    dseg = nifty.tools.makeDense(seg)

    # dseg.dtype, type(dseg)

    # "project to pixels"
    pixelData = nrag.projectScalarNodeDataToPixels(rag, dseg.astype('uint32'))
    # "done"
    #pixelDataF = numpy.require(pixelData, dtype='uint32',requirements='F')
    return pixelData





def makeSupervoxels(rawData, settings):
    
    sigma = settings['spSigmaHessian']
    edgeIndicator = vigra.filters.hessianOfGaussianEigenvalues(rawData, sigma)[:,:,0]
    seg, nseg = vigra.analysis.watershedsNew(edgeIndicator)

    # make smaller supervoxels via agglomerative clustering
    seg = makeSmallerSeg(seg, edgeIndicator, settings['reduceBy'], settings['sizeRegularizer'])








    return seg