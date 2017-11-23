import nifty
import nifty.graph.agglo
import nifty.segmentation
import numpy 
import h5py
import pylab

import scipy.sparse 









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


    mode = "test"

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


    # the data
    affF = "/home/tbeier/nice_probs/isbi_%s_offsetsV4_3d_meantda_damws2deval_final.h5"%mode
    affF = h5py.File(affF)
    aff = affF['data'][:,0:1,0:35,0:35]
    assert aff.shape[0] == n_offsets
    affF.close()
    shape = aff.shape[1:4]



    # load raw
    import skimage.io
    raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_%s.tif"%mode
    #raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_train.tif'
    raw = skimage.io.imread(raw_path)
    affF = "/home/tbeier/nice_probs/isbi_%s_offsetsV4_3d_meantda_damws2deval_final.h5"%mode
    raw = raw[0:1,0:35,0:35]
    #raw = numpy.rollaxis(raw ,0,3)


    shape = raw.shape



    def do_stuff(A):

        A_init = A.copy()
        Q = A.copy()
        I = scipy.sparse.identity(A.shape[0])

        for x in range(20):

            Q = A.dot(Q).dot(A.T) + I

        return Q


        

    g, affinities, offset_index = makeLongRangGridGraph(shape=shape,offsets=offsets, affinities=aff    )
    uv = g.uvIds()

    isLiftedEdge = offset_index.astype('int32')

    from scipy.sparse import coo_matrix
    n_pixels = shape[0]*shape[1]*shape[2]
    
    real_aff = 1.0 - affinities

    real_aff2 = numpy.concatenate([real_aff,real_aff])
    r = numpy.concatenate([uv[:,0], uv[:,1]])
    c = numpy.concatenate([uv[:,1], uv[:,0]])
    A = coo_matrix((real_aff2, (r, c)), shape=(n_pixels, n_pixels)) 
    from sklearn.preprocessing import normalize
    A = normalize(A, norm='l1', axis=0)
    A *= 0.9999


    B = do_stuff(A)
    print("get vals")
    vals = B[uv[:,0],uv[:,1]]
    vals = numpy.asarray(vals)
    print(type(vals))
    #vals = vals.todense()

    print(vals) 



    print("affminmax", vals.min(), vals.max())


    edgeSizes = numpy.ones(g.numberOfEdges, dtype='float32')
    nodeSizes = numpy.ones(g.numberOfNodes, dtype='float32')
    vals = numpy.require(vals, dtype='float32').squeeze()

    vals -= vals.min()
    vals /= vals.max()
    vals = 1.0 - vals
    print(vals.min(),vals.max())
    clusterPolicy = nifty.graph.agglo.liftedGraphEdgeWeightedClusterPolicy(graph=g,
        edgeIndicators=vals, edgeSizes=edgeSizes, isLiftedEdge=isLiftedEdge,  nodeSizes=nodeSizes)


    # run agglomerative clustering
    agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
    agglomerativeClustering.run(True,1)
    nodeSeg = agglomerativeClustering.result()


    nodeSeg = nodeSeg.reshape(shape)


    overlay = nifty.segmentation.segmentOverlay(raw[0,:,:],nodeSeg[0,:,:])

    pylab.imshow(nodeSeg[0,:,:])
    pylab.show()