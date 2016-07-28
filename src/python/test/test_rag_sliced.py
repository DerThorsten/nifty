import nifty
import vigra
import numpy
import os

chunked_rag = nifty.graph.rag.chunkedLabelsGridRagSliced
normal_rag  = nifty.graph.rag.explicitLabelsGridRag3D


def testExplicitLabelsRag3d():

    labels = [
        [
            [0,1],
            [0,0]
        ],
        [
            [1,1],
            [2,2]
        ],
        [
            [3,3],
            [3,3]
        ]
    ]

    labels = numpy.array(labels, dtype = 'uint32')

    vigra.writeHDF5(labels, "tmp.h5", "data", chunks = (2,2,1))

    ragA = chunked_rag("tmp.h5","data")
    ragB = normal_rag(labels)

    assert isinstance(ragA, nifty.graph.rag.ChunkedLabelsGridRagSliced)
    assert isinstance(ragB, nifty.graph.rag.ExplicitLabelsGridRag3D)

    shoudlEdges = [
        (0,1),
        (0,2),
        (1,2),
        (1,3),
        (2,3)

    ]

    shoudlNotEdges = [
       (0,3)
    ]


    assert ragA.numberOfNodes == 4
    assert ragB.numberOfNodes == 4

    assert ragA.numberOfEdges == len(shoudlEdges)
    assert ragB.numberOfEdges == len(shoudlEdges)

    edgeListA = []
    for edge in ragA.edges():
        edgeListA.append(edge)

    edgeListB = []
    for edge in ragB.edges():
        edgeListB.append(edge)

    assert len(edgeListA) == len(shoudlEdges)
    assert len(edgeListB) == len(shoudlEdges)


    for shouldEdge in shoudlEdges:

        fResA = ragA.findEdge(shouldEdge)
        fResB = ragB.findEdge(shouldEdge)
        assert fResA >= 0
        assert fResB >= 0
        uvA = ragA.uv(fResA)
        uvB = ragB.uv(fResB)
        uvA = sorted(uvA)
        uvB = sorted(uvB)
        assert uvA[0] == shouldEdge[0]
        assert uvA[1] == shouldEdge[1]
        assert uvB[0] == shouldEdge[0]
        assert uvB[1] == shouldEdge[1]

    for shouldNotEdge in shoudlNotEdges:

        fResA = ragA.findEdge(shouldNotEdge)
        fResB = ragB.findEdge(shouldNotEdge)
        assert fResA == -1
        assert fResB == -1

    os.remove("tmp.h5")


def compare_rags_from_files(labels_file, labels_key):

    with vigra.Timer("Chunked Nifty Rag"):
        rag_c = chunked_rag(labels_file, labels_key, numberOfThreads = 1)
    edges_c = rag_c.numberOfEdges
    print edges_c
    nodes_c = rag_c.numberOfNodes
    del rag_c


    labels = vigra.readHDF5(labels_file, labels_key).astype('uint32')

    with vigra.Timer("Nifty Rag"):
        rag_n = normal_rag(labels, numberOfThreads = -1 )
    edges_n = rag_n.numberOfEdges
    nodes_n = rag_n.numberOfNodes

    with vigra.Timer("Vigra Rag"):
        rag_v = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(labels.shape), labels)
    nodes_v = rag_v.nodeNum
    edges_v = rag_v.edgeNum

    assert nodes_c == nodes_n, str(nodes_c) + " , " + str(nodes_n)
    #assert nodes_v == nodes_n, str(nodes_v) + " , " + str(nodes_n)

    assert edges_c == edges_n, str(edges_c) + " , " + str(edges_n)
    assert edges_v == edges_n, str(edges_v) + " , " + str(edges_n)

    print "Checks out"




if __name__ == '__main__':

    testExplicitLabelsRag3d()

    #files = ["/home/consti/Work/projects/phd_prototyping/vigra_chunked/test_labels_chunked.h5",
    #        "/home/consti/Work/data_neuro/test_block/sliced_data/test-seg.h5",
    #        "/home/consti/Work/data_neuro/large_tests/sampleA/sampleA_ws.h5"]

    #for f in files:

    #    print f
    #    test_rag_build(f, "data")
