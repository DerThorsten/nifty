from __future__ import print_function
import nifty
import numpy
import nifty
import tempfile
import shutil
import os 

from nose.tools import assert_almost_equals

nrag = nifty.graph.rag


def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def testAccumulateEdgeMeanAndLengthRag2d():

    labels = numpy.array( [
        [0,1,2],
        [0,0,2],
        [3,3,2],
        [4,4,4]
    ]).T

    data = numpy.array([
        [1,1,1],
        [1,1,1],
        [1,1,1],
        [1,1,1]
    ], dtype='float32').T

    rag = nrag.gridRag(labels)

    features = nrag.accumulateEdgeMeanAndLength(rag, data, [2,2], 1)


    edgesAndCounts = [
        ((0,1), 2*2 ),
        ((0,2), 1*2 ),
        ((0,3), 2*2 ),
        ((1,2), 1*2 ),
        ((2,3), 1*2 ),
        ((2,4), 1*2 ),
        ((3,4), 2*2 )
    ]

    mean = features[:,0]
    count = features[:,1]

    for edge,c in edgesAndCounts:

        edgeId = rag.findEdge(edge[0], edge[1])

        print (edge, "myc",count[edgeId],"shouldc",c)
        assert count[edgeId] == c



def testAccumulateMeanAndLengthRag2d():

    labels = numpy.array( [
        [0,  1, 2],
        [0,  0, 2],
        [3,  3, 2],
        [4,  4, 4]
    ]).T

    data = numpy.array([
        [1,  2, 3],
        [4,  5, 6],
        [7,  8, 9],
        [10,11,12]
    ], dtype='float32').T

    rag = nrag.gridRag(labels)

    eFeatures, nFeatures = nrag.accumulateMeanAndLength(rag, data, [2,2], 1)


    edgesAndCountsAndMeans = [
        ((0,1), 2*2, float(1+2+2+5)/4 ),
        ((0,2), 1*2, float(5+6)/2 ),
        ((0,3), 2*2, float(4+5+7+8)/4 ),
        ((1,2), 1*2, float(2+3)/2 ),
        ((2,3), 1*2, float(8+9)/2 ),
        ((2,4), 1*2, float(9+12)/2 ),
        ((3,4), 2*2, float(7+8+10+11)/4 )
    ]

    nodeAndCountsAndMeans = [
        (0,3, (1.0+4.0+5.0)/3.0    ),
        (1,1, (2.0)/1.0            ),
        (2,3, (3.0+6.0+9.0)/3.0    ),
        (3,2, (7.0+8.0)/2.0        ),
        (4,3, (10.0+11.0+12.0)/3.0 )
    ]

    eMean = eFeatures[:,0]
    eCount = eFeatures[:,1]

    nMean = nFeatures[:,0]
    nCount = nFeatures[:,1]

    for edge,c,m in edgesAndCountsAndMeans:
        edgeId = rag.findEdge(edge[0], edge[1])
        assert eCount[edgeId] == c
        assert_almost_equals(m, eMean[edgeId],5)



    for node,c,m in nodeAndCountsAndMeans:
        assert nCount[node] == c
    for node,c,m in nodeAndCountsAndMeans:
        assert_almost_equals(m, nMean[node],5)