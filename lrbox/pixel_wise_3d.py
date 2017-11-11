import nifty
import nifty.graph.agglo
import nifty.segmentation
import numpy 
import h5py
import pylab











def makeLongRangGridGraph(shape, offsets, affinities):
    shape = list(shape)

    if len(shape) == 2:
        raise NotImplementedError
    else:
        g = nifty.graph.UndirectedGraph()
        g_aff, g_offset_index = nifty.graph.longRangeGridGraph3D(
            g,
            shape,
            numpy.require(offsets, dtype='int64'),
            numpy.require(affinities, dtype='float32')
        )

        return g, g_aff, g_offset_index 







if __name__ == "__main__":


    # the offsets
    offsets = numpy.array([
    [-1, 0, 0], [0, -1, 0], [0, 0, -1],                  # direct 3d nhood for attractive edges
    [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],  # indirect 3d nhood for dam edges
    [0, -9, 0], [0, 0, -9],                  # long range direct hood
    [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],  # inplane diagonal dam edges
    [0, -27, 0], [0, 0, -27]
    ]).astype('int64')
    n_offsets = offsets.shape[0]
    print("offsets", offsets.shape)


    # the test data
    affF = "/home/tbeier/nice_probs/isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5"
    affF = h5py.File(affF)
    aff = affF['data'][:,10:16,0:512,0:512]
    assert aff.shape[0] == n_offsets
    affF.close()
    shape = aff.shape[1:4]



    # load raw
    import skimage.io
    raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_test.tif"
    #raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_test.tif'
    raw = skimage.io.imread(raw_path)
    raw = raw[10:16,0:512,0:512]
    #raw = numpy.rollaxis(raw ,0,3)




    print("aff",aff.shape)
    print("shape",shape)
    # sanity check
    if False:
        pylab.imshow(aff[0,0,:,:])
        pylab.show()



    

    g, affinities, offset_index = makeLongRangGridGraph(shape=shape,offsets=offsets, affinities=aff    )
    print(g,affinities.shape, offset_index.shape)
    isLiftedEdge = offset_index.astype('uint8')

    w = numpy.where(offset_index<=2)
    isLiftedEdge[:] = 1
    isLiftedEdge[w] = 0 


    edgeSizes = numpy.ones(g.numberOfEdges, dtype='float32')
    nodeSizes = numpy.ones(g.numberOfNodes, dtype='float32')
    aff = numpy.require(aff, dtype='float32')


    print("affminmax", affinities.min(), affinities.max())

    clusterPolicy = nifty.graph.agglo.liftedGraphEdgeWeightedClusterPolicy(graph=g,
        edgeIndicators=affinities, edgeSizes=edgeSizes, isLiftedEdge=isLiftedEdge,  nodeSizes=nodeSizes)


    # run agglomerative clustering
    agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(True,1000)
    nodeSeg = agglomerativeClustering.result()


    nodeSeg = nodeSeg.reshape(shape)



    pylab.imshow(nifty.segmentation.segmentOverlay(raw[2,:,:], 
        nifty.segmentation.connectedComponents(nodeSeg[2,:,:]),beta=0.15, thin=False))
    pylab.show()