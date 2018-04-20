import nifty
import nifty.graph.opt.ho_multicut as homc
import nifty.segmentation as nseg
import nifty.graph.rag as nrag
import nifty.graph.agglo as nagglo
import pylab
import nifty.cgp as ncgp
import numpy
import scipy.spatial.distance
import nifty.cgp as ncgp
import vigra
import math
import sklearn.neighbors

def showseg(rgb, seg):
    visu = nseg.segmentOverlay(rgb, seg)
    pylab.imshow(visu)
    pylab.show()

def make_overseg(rgb):

    lab =  vigra.colors.transform_RGB2Lab(rgb)
    gmag =vigra.filters.gaussianGradientMagnitude(lab, 3.0)
    overseg, nsegments = vigra.analysis.watershedsNew(gmag)
    
    overseg = overseg.squeeze()
    overseg -= overseg.min()    
    overseg = overseg.astype('uint64')
    gmag = gmag.squeeze()



    rag = nrag.gridRag(overseg)


    gmag = gmag.view(numpy.ndarray)

    print(gmag.shape, overseg.shape)

    print("..")
    edge_features, node_features = nrag.accumulateMeanAndLength(
        rag, gmag)

    meanEdgeStrength = edge_features[:,0]
    edgeSizes = edge_features[:,1]
    nodeSizes = node_features[:,1]

    # cluster-policy  
    clusterPolicy = nagglo.edgeWeightedClusterPolicy(
        graph=rag, edgeIndicators=meanEdgeStrength,
        edgeSizes=edgeSizes, nodeSizes=nodeSizes,
        numberOfNodesStop=300, sizeRegularizer=1.6)

    # run agglomerative clustering
    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run()
    nodeSeg = agglomerativeClustering.result()
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)
    seg = vigra.analysis.labelImage(seg.astype('uint32'))
    seg -= seg.min()


    return seg.astype('uint64')





def makePotts(d):

    # similarity
    s = math.exp(-1.0*d)

    a = numpy.zeros([2,2])

    a[1,0] = s
    a[0,1] = s

    return a

def computeFeatures(rgb, rag):

    uv = rag.uvIds()
    nrag = nifty.graph.rag

    # list of all edge features we fill 
    feats = []

    # helper function to convert 
    # node features to edge features
    def nodeToEdgeFeat(nodeFeatures):
        uF = nodeFeatures[uv[:,0], :]
        vF = nodeFeatures[uv[:,1], :]
        feats = [ numpy.abs(uF-vF), uF + vF, uF *  vF,
                 numpy.minimum(uF,vF), numpy.maximum(uF,vF)]
        return numpy.concatenate(feats, axis=1)

    for c in range(3):
        # accumulate features from raw data
        gray = rgb[:,:,c] 
        channels_raw = [
            gray,
            vigra.filters.gaussianGradientMagnitude(gray, 0.5).squeeze(),
            #vigra.filters.gaussianGradientMagnitude(gray, 1.5).squeeze(),
            #vigra.filters.hessianOfGaussianEigenvalues(gray, 0.3),
            #vigra.filters.hessianOfGaussianEigenvalues(gray, 0.6),
            #vigra.filters.structureTensorEigenvalues(gray, 0.3, 1.0),
            vigra.filters.structureTensorEigenvalues(gray, 0.6, 1.5),
            vigra.filters.structureTensorEigenvalues(gray, 1.6, 3.5),
        ]
        channels = []
        for channel in channels_raw:
            if channel.ndim == 2:
                channels.append(channel)
            elif channel.ndim == 3:
                for c in range(channel.shape[2]):
                    channels.append(channel[...,c])

        
        for channel in channels:
            mi = float(channel.min())
            channel -= mi 
            ma = float(channel.max())
            channel /= ma

            fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=rgb[:,:,c],
                minVal=0.0,  maxVal=1.0, numberOfThreads=1)
            feats.append(fRawEdge)
            #feats.append(nodeToEdgeFeat(fRawNode))


    # accumulate node and edge features from
    # superpixels geometry 
    #fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    #feats.append(fGeoEdge)

    #fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    #feats.append(nodeToEdgeFeat(fGeoNode))

    return numpy.concatenate(feats, axis=1)




f = "/home/tbeier/datasets/BSR/BSDS500/data/images/val/21077.jpg"
rgb  = vigra.impex.readImage(f).astype('float32')


rgb = numpy.swapaxes(rgb, 0,1)

overseg = make_overseg(rgb)
#showseg(rgb, overseg)
rag = nrag.gridRag(overseg)


gmag =vigra.filters.gaussianGradientMagnitude(rgb, 3.0)

edge_features, node_features = nrag.accumulateMeanAndLength(
        rag, gmag.squeeze())
meanEdgeStrength = edge_features[:,0]
edgeSizes = edge_features[:,1]

w = numpy.exp(-0.1*meanEdgeStrength) - 0.3

print(w.min(), w.max())
#w *=-1.0



#w = numpy.random.rand(rag.numberOfEdges) - 0.5

# unaries are added direct all at once
obj = homc.hoMulticutObjective(rag, w)



eFeat = computeFeatures(rgb=rgb, rag=rag)
eFeat -= numpy.min(eFeat,axis=1)[:,None]
eFeat /= numpy.max(eFeat,axis=1)[:,None]



knn = sklearn.neighbors.NearestNeighbors(n_neighbors=20)
knn.fit(eFeat)

potts =  numpy.zeros([2,2])
potts[1,0] = 1
potts[0,1] = 1



bla = set()
c = 0
for e in range(rag.numberOfEdges):
    dist, ind  = knn.kneighbors(eFeat[e,:][None,:], return_distance=True)
    dist = dist.squeeze()
    ind = ind.squeeze()

    #print("dist",dist.shape, "ind",ind.shape)
    
    for e2,d in zip(ind,dist):

        #print(ind,e2)
        if e2 < e:

            key = tuple([min(e,e2),max(e2,e)])
            if key not in bla:
                #print("E",e, "E2",e2, "D",d)
                cost = numpy.exp(-30.001*d)
                if c < 8:
                    print("d",d,"cost",cost)
                c +=1
                obj.addHigherOrderFactor(potts*cost*1.0 , [e, e2])
                bla.add(key)





if True:

    factory = obj.hoMulticutIlpFactory(
        integralHo=False,
        ilp=True,
        timeLimit=800.0,
        ilpSolverSettings=nifty.graph.opt.ho_multicut.ilpSettings(
            relativeGap=0.0,
            #timeLimit=3
        )
    )
else:

    ilpFactory = obj.hoMulticutIlpFactory(
        integralHo=False,
        ilp=True,
        ilpSolverSettings=nifty.graph.opt.ho_multicut.ilpSettings(
            relativeGap=0.1,
            timeLimit=5.0
        )
    )
    obj.fusionMoveSettings(hoMcFactory=ilpFactory)
    factory = obj.hoMulticutDualDecompositionFactory(
        crfSolver='graphcut',
        numberOfIterations=100, stepSize=1.1,
        absoluteGap=0.1)


solver = factory.create(obj)
arg = solver.optimize(obj.verboseVisitor())



seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)
seg = vigra.analysis.labelImage(seg.astype('uint32'))
seg -= seg.min()

showseg(rgb, seg)

