import nifty
import numpy
import skimage

g =  nifty.graph.UndirectedGraph(4)
edges =  numpy.array([[0,1],[0,2],[0,3]],dtype='float')
g.insertEdges(edges)
print g




rag3d = nifty.graph.Rag3d()
print rag3d
