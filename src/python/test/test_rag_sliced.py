import nifty
import vigra
import numpy

chunked_rag = nifty.graph.rag.chunkedLabelsGridRagSliced
normal_rag  = nifty.graph.rag.explicitLabelsGridRag3D

def test_rag_build(labels_file, labels_key):
    rag_c = chunked_rag(labels_file, labels_key)
    print rag_c.numberOfEdges





if __name__ == '__main__':
    labels_file = "/home/constantin/Work/my_projects/phd_prototyping/vigra_chunked/test_labels_chunked.h5"
    labels_key  = "data"

    test_rag_build(labels_file, labels_key)
