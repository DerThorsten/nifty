import nifty
import nifty.graph.agglo
import nifty.segmentation
import h5py
import numpy
import pylab
import vigra
# load affinities
z = 5
w = 512
path_affinities = "/home/tbeier/nice_probs/isbi_test_default.h5"
offsets = numpy.array([
    [-1,0],[0,-1],
    [-9,0],[0,-9],[-9,-9],[9,-9],
    [-9,-4],[-4,-9],[4,-9],[9,-4],
    [-27,0],[0,-27],[-27,-27],[27,-27]
]).astype('int64')
print(offsets.shape)

f5_affinities = h5py.File(path_affinities)
affinities = f5_affinities['data']
affinities = affinities[:,z,0:w, 0:w]
affinities = numpy.rollaxis(affinities ,0,3)
affinities = numpy.require(affinities, dtype='float32', requirements=['C'])

# load raw
import skimage.io
raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_test.tif"
#raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif'
raw = skimage.io.imread(raw_path)
raw = raw[z,0:w,0:w]


isMergeEdgeOffset = numpy.zeros(offsets.shape[0], dtype='bool')
isMergeEdgeOffset[0:2] = True


if True:
    affinities = vigra.gaussianSmoothing(vigra.taggedView(affinities,'xyc'),0.2)
    with nifty.Timer("jay"):
        nodeSeg = nifty.graph.agglo.pixelWiseFixation2D(
            mergePrios= (1.0 - affinities),         #vigra.gaussianSmoothing(vigra.taggedView( (1.0 - affinities),'xyc'),0.01),
            notMergePrios=affinities,               #vigra.gaussianSmoothing(vigra.taggedView( (affinities),'xyc'),0.01),
            offsets=offsets,
            isMergeEdgeOffset=isMergeEdgeOffset
        )

    #nodeSeg = nodeSeg.reshape(shape)

    import pylab

    pylab.imshow(nodeSeg)
    pylab.show()
    pylab.imshow(nifty.segmentation.segmentOverlay(raw, nodeSeg, showBoundaries=False))
    pylab.show()
    pylab.imshow(nifty.segmentation.markBoundaries(raw, nodeSeg, color=(1,0,0)))
    pylab.show()


else:

    pylab.imshow(affinities[:,:,0])
    pylab.show()

    shape = affinities.shape[0:2]
    n_offsets = affinities.shape[2]


    def vi(x,y):
        return x*shape[1] + y



    g = nifty.graph.undirectedGraph(shape[0]*shape[1])

    print("setup graph")
    # setup graph
    for x in range(shape[0]):
        for y in range(shape[1]):
            u = vi(x,y)
            for offset_index in range(n_offsets):
                ox,oy = offsets[offset_index,:]
                # bounds check
                if x+ox >=0 and x+ox < shape[0] and y+oy >=0 and y+oy < shape[1]:
                    v = vi(x+ox, y+oy)
                    g.insertEdge(u,v)  
                    #print(g.numberOfEdges)

    is_merge_edge   = numpy.zeros(g.numberOfEdges,dtype='uint8')
    merge_prios     = numpy.zeros(g.numberOfEdges,dtype='float32')
    not_merge_prios = numpy.zeros(g.numberOfEdges,dtype='float32')
    edge_sizes      =     numpy.ones(g.numberOfEdges,dtype='float32')

    print("setup weights")
    for x in range(shape[0]):
        for y in range(shape[1]):
            u = vi(x,y)
            for offset_index in range(n_offsets):
                ox,oy = offsets[offset_index,:]
                # bounds check
                if x+ox >=0 and x+ox < shape[0] and y+oy >=0 and y+oy < shape[1]:
                    v = vi(x+ox, y+oy)
                    edge = g.findEdge(u,v)
                    p_cut = affinities[x,y,offset_index]
                    not_merge_prios[edge] = p_cut
                    merge_prios[edge] =  1.0 - p_cut
                    if abs(ox) + abs(oy) == 1:
                        is_merge_edge[edge] = 1


    print("do agglo")
    # cluster-policy  
    clusterPolicy = nifty.graph.agglo.fixationClusterPolicy(
        graph=g, mergePrios=merge_prios,notMergePrios=not_merge_prios,
        isMergeEdge=is_merge_edge, edgeSizes=edge_sizes,
        numberOfNodesStop=1)


    # run agglomerative clustering
    agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(verbose=0)
    print("done")
    nodeSeg = agglomerativeClustering.result()
    nodeSeg = nodeSeg.reshape(shape)

    import nifty.segmentation


    import pylab

    pylab.imshow(nodeSeg)
    pylab.show()



    pylab.imshow(nifty.segmentation.segmentOverlay(raw, nodeSeg, showBoundaries=False))
    pylab.show()


    pylab.imshow(nifty.segmentation.markBoundaries(raw, nodeSeg, color=(1,0,0)))
    pylab.show()