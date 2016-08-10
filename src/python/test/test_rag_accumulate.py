from __future__ import print_function
import nifty
import numpy
import nifty
import tempfile
import shutil
import os 

nrag = nifty.graph.rag


def ensureDir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def testExplicitLabelsRag2d():

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

