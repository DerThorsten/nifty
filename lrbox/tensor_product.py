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
    aff = affF['data'][:,0:1,0:200,0:200]
    #assert aff.shape[0] == n_offsets
    affF.close()
    shape = aff.shape[1:4]



    # load raw
    import skimage.io
    raw_path = "/home/tbeier/src/nifty/src/python/examples/multicut/NaturePaperDataUpl/ISBI2012/raw_%s.tif"%mode
    #raw_path = '/home/tbeier/src/nifty/mysandbox/NaturePaperDataUpl/ISBI2012/raw_train.tif'
    raw = skimage.io.imread(raw_path)
    affF = "/home/tbeier/nice_probs/isbi_%s_offsetsV4_3d_meantda_damws2deval_final.h5"%mode
    raw = raw[0:1,0:200,0:200]
    #raw = numpy.rollaxis(raw ,0,3)


    shape = raw.shape



    def do_stuff(A):
        #return A
        A_init = A.copy()
        Q = A.copy()
        I = scipy.sparse.identity(A.shape[0])
        AT = A.T
        for x in range(1):
            print("x",x)

            AQ = A.dot(Q)
            #AQ.eliminate_zeros()
            #AQ = AQ[AQ.getnnz(1)>0.0001]
            Q = AQ.dot(AT)  + I
            #Q.eliminate_zeros()
            #Q = Q[Q.getnnz(1)>0.0001]
        return Q


    pylab.imshow(raw[0,:,:])
    #pylab.show()

    g, affinities, offset_index = makeLongRangGridGraph(shape=shape,offsets=offsets, affinities=aff    )
    uv = g.uvIds()
    u = uv[:,0]
    v = uv[:,1]

    isLiftedEdge = offset_index.astype('int32')
    w = numpy.where(offset_index<=2)
    isLiftedEdge[:] = 1
    isLiftedEdge[w] = 0 

    from scipy.sparse import coo_matrix
    n_pixels = shape[0]*shape[1]*shape[2]
    
    real_aff = 1.0 - affinities

    # non_zero = numpy.where(real_aff>0.00001)[0]
    # real_aff = real_aff[non_zero]
    # u = u[non_zero]
    # v = v[non_zero]

    real_aff2 = numpy.concatenate([real_aff,real_aff])
    r = numpy.concatenate([u,v])
    c = numpy.concatenate([v,u])

    A = coo_matrix((real_aff2, (r, c)), shape=(n_pixels, n_pixels)) 
    from sklearn.preprocessing import normalize
    A = normalize(A, norm='l1', axis=0)
    A *= 0.99999999
    B = do_stuff(A)


    non_lifted_e = numpy.where(isLiftedEdge==False)[0]
    nlu = u[non_lifted_e]
    nlv = v[non_lifted_e]
    non_lifted_uv = numpy.array((nlu,nlv)).T
    if True:
        new_uv = B.nonzero()
        #print(new_uv).shape()
        vals = B[new_uv]
        vals = numpy.asarray(vals).squeeze()
        new_g = nifty.graph.UndirectedGraph(n_pixels)

        print(vals.shape, new_uv[0].shape)

        where = numpy.where(new_uv[0] <  new_uv[1])[0 ]
        new_uv =new_uv[0][where],new_uv[1][where]
        vals = vals[where]

        new_g.insertEdges(numpy.array(new_uv).T)
        
        edgeSizes = numpy.ones(new_g.numberOfEdges, dtype='float32')
        nodeSizes = numpy.ones(new_g.numberOfNodes, dtype='float32')
        isLiftedEdge = numpy.ones(new_g.numberOfEdges, dtype="int")

        print("non_lifted_uv",non_lifted_uv.shape,non_lifted_uv.dtype)
        e = new_g.findEdges(non_lifted_uv.astype('int'))
        isLiftedEdge[e] = False

        vals = numpy.require(vals, dtype='float32').squeeze()

        vals -= vals.min()
        vals /= vals.max()
        vals = 1.0 - vals



        print(vals.min(),vals.max())
        clusterPolicy = nifty.graph.agglo.liftedGraphEdgeWeightedClusterPolicy(graph=new_g,
            edgeIndicators=vals, edgeSizes=edgeSizes, isLiftedEdge=isLiftedEdge,  nodeSizes=nodeSizes,
            stopConditionType="numberOfNodes",stopNodeNumber=100)

        # run agglomerative clustering
        agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
        agglomerativeClustering.run(True,10000)
        nodeSeg = agglomerativeClustering.result()

    else:

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
            edgeIndicators=vals, edgeSizes=edgeSizes, isLiftedEdge=isLiftedEdge,  nodeSizes=nodeSizes,
            stopConditionType="numberOfNodes",stopNodeNumber=100)


        # run agglomerative clustering
        agglomerativeClustering = nifty.graph.agglo.agglomerativeClustering(clusterPolicy) 
        agglomerativeClustering.run(True,10000)
        nodeSeg = agglomerativeClustering.result()


    nodeSeg = nodeSeg.reshape(shape)


    overlay = nifty.segmentation.segmentOverlay(raw[0,:,:],nodeSeg[0,:,:])

    pylab.imshow(overlay)
    pylab.show()