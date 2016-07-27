import nifty
import vigra
import numpy

chunked_rag = nifty.graph.rag.chunkedLabelsGridRagSliced
normal_rag  = nifty.graph.rag.explicitLabelsGridRag3D

def test_rag_build(labels_file, labels_key):

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

    files = ["/home/consti/Work/projects/phd_prototyping/vigra_chunked/test_labels_chunked.h5",
            "/home/consti/Work/data_neuro/test_block/sliced_data/test-seg.h5",
            "/home/consti/Work/data_neuro/large_tests/sampleA/sampleA_ws.h5"]

    for f in files:

        print f
        test_rag_build(f, "data")
