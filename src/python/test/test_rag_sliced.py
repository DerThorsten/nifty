import nifty
import vigra
import numpy

chunked_rag = nifty.graph.rag.chunkedLabelsGridRagSliced
normal_rag  = nifty.graph.rag.explicitLabelsGridRag3D

def test_rag_build(labels_file, labels_key):
    rag_c = chunked_rag(labels_file, labels_key)
    edges_c = rag_c.numberOfEdges
    nodes_c = rag_c.numberOfNodes
    del rag_c

    labels = vigra.readHDF5(labels_file, labels_key)

    rag_n = normal_rag(labels)
    edges_n = rag_n.numberOfEdges
    nodes_n = rag_n.numberOfNodes

    rag_v = vigra.graphs.regionAdjacencyGraph(vigra.graphs.gridGraph(labels.shape), labels)
    nodes_v = rag_v.nodeNum
    edges_v = rag_v.edgeNum

    assert nodes_c == nodes_n, str(nodes_c) + " , " + str(nodes_n)
    assert nodes_v == nodes_n, str(nodes_v) + " , " + str(nodes_n)

    print "Vigra edges:", edges_v
    assert edges_c == edges_n, str(edges_c) + " , " + str(edges_n)
    assert edges_v == edges_n, str(edges_v) + " , " + str(edges_n)


    print "Check"





if __name__ == '__main__':
    labels_file = "/home/consti/Work/projects/phd_prototyping/vigra_chunked/test_labels_chunked.h5"
    labels_key  = "data"

    test_rag_build(labels_file, labels_key)
