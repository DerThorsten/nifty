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
        numberOfNodesStop=20, sizeRegularizer=0.2)

    # run agglomerative clustering
    agglomerativeClustering = nagglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run()
    nodeSeg = agglomerativeClustering.result()
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, nodeSeg)
    seg = vigra.analysis.labelImage(seg.astype('uint32'))
    seg -= seg.min()


    return seg.astype('uint64')

def make_j_prio(v2=0.0, v3=1.0, v4=1.0):

    vt3 = numpy.zeros([2,2,2])
    vt4 = numpy.zeros([2,2,2,2])

    for x0 in range(2):
        for x1 in range(2):
            for x2 in range(2):
                for x3 in range(2): 
                    s = x0 + x1 + x2 + x3

                    if s == 0:
                        vt4[x0, x1, x2, x3] = 0.0
                    if s == 2:
                        vt4[x0, x1, x2, x3] = v2
                    if s == 3:
                        vt4[x0, x1, x2, x3] = v3
                    if s == 4:
                        vt4[x0, x1, x2, x3] = v3

    for x0 in range(2):
        for x1 in range(2):
            for x2 in range(2):

                s = x0 + x1 + x2 

                if s == 0:
                    vt3[x0, x1, x2] = 0.0
                if s == 2:
                    vt3[x0, x1, x2] = v2
                if s == 3:
                    vt3[x0, x1, x2] = v3
                if s == 4:
                    vt3[x0, x1, x2] = v3

    return vt3, vt4

def find_j_edges(rag, overseg):
    assert overseg.min() == 0 

    tgrid = ncgp.TopologicalGrid2D(overseg+1)

    bounds = tgrid.extractCellsBounds()
    bounds0 = bounds[0]
    bounds1 = bounds[1]
    print(bounds[0])



    def cell1ToEdge(cell1Label):
        assert cell1Label >=1
        a,b = numpy.array(bounds1[int(cell1Label)-1])
        assert a>=1 
        assert b>=1
        return rag.findEdge(a-1, b-1)

    res = []
    for i in range(tgrid.numberOfCells[0]):
        
        cell1Labels = numpy.array(bounds0[i])
        edges = [cell1ToEdge(c) for c in cell1Labels]

        res.append(edges)


    return res

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
        fRawEdge, fRawNode = nrag.accumulateStandartFeatures(rag=rag, data=rgb[:,:,c],
            minVal=0.0, maxVal=255.0, numberOfThreads=1)
        feats.append(fRawEdge)
        feats.append(nodeToEdgeFeat(fRawNode))


    # accumulate node and edge features from
    # superpixels geometry 
    fGeoEdge = nrag.accumulateGeometricEdgeFeatures(rag=rag, numberOfThreads=1)
    feats.append(fGeoEdge)

    fGeoNode = nrag.accumulateGeometricNodeFeatures(rag=rag, numberOfThreads=1)
    feats.append(nodeToEdgeFeat(fGeoNode))

    return numpy.concatenate(feats, axis=1)




f = "/home/tbeier/datasets/BSR/BSDS500/data/images/train/66075.jpg"
rgb  = vigra.impex.readImage(f).astype('float32')


rgb = numpy.swapaxes(rgb, 0,1)

overseg = make_overseg(rgb)
showseg(rgb, overseg)
rag = nrag.gridRag(overseg)


gmag =vigra.filters.gaussianGradientMagnitude(rgb, 3.0)

edge_features, node_features = nrag.accumulateMeanAndLength(
        rag, gmag.squeeze())
meanEdgeStrength = edge_features[:,0]
edgeSizes = edge_features[:,1]

w = numpy.exp(-0.1*meanEdgeStrength) - 0.6

print(w.min(), w.max())
#w *=-1.0

print("weights",w)

#w = numpy.random.rand(rag.numberOfEdges) - 0.5

# unaries are added direct all at once
obj = homc.hoMulticutObjective(rag, w)



eFeat = computeFeatures(rgb=rgb, rag=rag)
eFeat -= numpy.min(eFeat,axis=1)[:,None]
eFeat /= numpy.max(eFeat,axis=1)[:,None]



knn = sklearn.neighbors.NearestNeighbors(n_neighbors=5)
knn.fit(eFeat)

potts =  numpy.zeros([2,2])
potts[1,0] = 0.1
potts[0,1] = 0.1



bla = set()
for e in range(rag.numberOfEdges):
    d = knn.kneighbors(eFeat[e,:][None,:], return_distance=False)
    d = d[d!=e]

    
    for e2 in d:
        key = tuple([min(e,e2),max(e2,e)])
        if key not in bla:

            obj.addHigherOrderFactor(potts, [e, e2])
            bla.add(key)



# for e_a in range(rag.numberOfEdges):
#     for e_b in range(e_a+1, rag.numberOfEdges):

#         if e_a != e_b:

#             d = pdist[e_a, e_b]
#             if d < 0.05:
#                 vt = makePotts(d)
#                 obj.addHigherOrderFactor(vt, [e_a, e_b])
#                 print("d",pdist[e_a, e_b])


#sys.exit()


vt3,vt4 = make_j_prio(v3=0.3, v4=0.2)

for edges in find_j_edges(rag=rag, overseg=overseg):

    if len(edges) == 3:
        obj.addHigherOrderFactor(vt3, edges)

    if len(edges) == 4:
        obj.addHigherOrderFactor(vt4, edges)





factory = obj.hoMulticutIlpFactory()
solver = factory.create(obj)
arg = solver.optimize(obj.verboseVisitor())



seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, arg)
seg = vigra.analysis.labelImage(seg.astype('uint32'))
seg -= seg.min()

showseg(rgb, seg)

